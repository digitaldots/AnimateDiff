import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch
import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from AnimateDiff.animatediff.models.unet import UNet3DConditionModel
from AnimateDiff.animatediff.pipelines.pipeline_animation import AnimationPipeline
from AnimateDiff.animatediff.utils.util import save_videos_grid
from AnimateDiff.animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from AnimateDiff.animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv
import pdb
import glob
from safetensors import safe_open
import math
from pathlib import Path


def main(**kwargs):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    inference_config = OmegaConf.load(kwargs['inference_config'])

    config = OmegaConf.load(kwargs['config'])

    for model_idx, (config_key, model_config) in enumerate(list(config.items())):

        motion_modules = 'AnimateDiff/models/Motion_Module/mm_sd_v15.ckpt'
        motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
        for motion_module in motion_modules:

            ### >>> create validation pipeline >>> ###
            tokenizer = CLIPTokenizer.from_pretrained(kwargs['pretrained_model_path'], subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(kwargs['pretrained_model_path'], subfolder="text_encoder")
            vae = AutoencoderKL.from_pretrained(kwargs['pretrained_model_path'], subfolder="vae")
            unet = UNet3DConditionModel.from_pretrained_2d(kwargs['pretrained_model_path'],
                                                           subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(
                                                               inference_config.unet_additional_kwargs))

            if is_xformers_available():
                unet.enable_xformers_memory_efficient_attention()
            else:
                assert False

            pipeline = AnimationPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            ).to("cuda")

            # 1. unet ckpt
            # 1.1 motion module
            motion_module_state_dict = torch.load(motion_module, map_location="cuda")
            if "global_step" in motion_module_state_dict:
                func_args.update({"global_step": motion_module_state_dict["global_step"]})
            missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
            assert len(unexpected) == 0

            # 1.2 T2I
            model_config.path = 'AnimateDiff/models/DreamBooth_LoRA/lyriel_v16.safetensors'
            if model_config.path != "":
                if model_config.path.endswith(".ckpt"):
                    state_dict = torch.load(model_config.path)
                    pipeline.unet.load_state_dict(state_dict)

                elif model_config.path.endswith(".safetensors"):
                    state_dict = {}
                    with safe_open(model_config.path, framework="pt", device="cuda") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)

                    is_lora = all("lora" in k for k in state_dict.keys())
                    if not is_lora:
                        base_state_dict = state_dict
                    else:
                        base_state_dict = {}
                        with safe_open(model_config.base, framework="pt", device="cuda") as f:
                            for key in f.keys():
                                base_state_dict[key] = f.get_tensor(key)

                    # vae
                    converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
                    pipeline.vae.load_state_dict(converted_vae_checkpoint)
                    # unet
                    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
                    pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
                    # text_model
                    pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)

                    # import pdb
                    # pdb.set_trace()
                    if is_lora:
                        pipeline = convert_lora(pipeline, state_dict, alpha=model_config.lora_alpha)

            pipeline.to("cuda")
    return pipeline
