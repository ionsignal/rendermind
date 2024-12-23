import os
import sys
import gc
import glob

import imageio_ffmpeg
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

# Add the parent directory of hyvideo to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import comfy.model_management as mm
from comfy.utils import load_torch_file
from comfy.model_base import BaseModel, ModelType
from comfy.latent_formats import LatentFormat
from comfy.model_patcher import ModelPatcher

from diffusers.video_processor import VideoProcessor

from hyvideo.diffusion.pipelines import HunyuanVideoPipeline
from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
from hyvideo.modules.models import HYVideoDiffusionTransformer
from hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from hyvideo.utils.data_utils import align_to
from hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from hyvideo.constants import PROMPT_TEMPLATE
from hyvideo.text_encoder import TextEncoder

# Configuration
MODEL_PATH = "/home/ubuntu/share/comfyui/models/diffusion_models/hunyuan-video-720-cfgdistill-fp8-e4m3fn.safetensors"
VAE_PATH = "/home/ubuntu/share/comfyui/models/vae/hunyuan-video-vae-bf16.safetensors"
VAE_CONFIG = "/home/ubuntu/share/comfyui/custom_nodes/comfyui-hunyuan-video/configs/hy_vae_config.json"
LLM_PATH = "/home/ubuntu/share/comfyui/models/llm/llava-llama-3-8b-text-encoder-tokenizer"
CLIP_PATH = "/home/ubuntu/share/comfyui/models/clip/clip-vit-large-patch14"
PROMPT = "high quality nature video of a excited brown bear running down a stream, masterpiece, best quality"
NEGATIVE_PROMPT = "bad quality video"
INPUT_FRAMES_PATH = "/home/ubuntu/share/tests-frames"
OUTPUT_VIDEO = "output_video.mp4"
WIDTH = 640
HEIGHT = 368
NUM_FRAMES = 49
STEPS = 30
EMBEDDED_GUIDANCE_SCALE = 6.0
FLOW_SHIFT = 9.0
SEED = 0
DENOISE_STRENGTH = 0.85
FORCE_OFFLOAD = False
BASE_DTYPE = torch.bfloat16
QUANT_DTYPE = torch.float8_e4m3fn
PARAMS_TO_KEEP = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}

class HyVideoModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = {}

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v

class HyVideoModelConfig:
    def __init__(self, dtype):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = LatentFormat()
        self.latent_format.latent_channels = 16
        self.manual_cast_dtype = dtype
        self.sampling_settings = {"multiplier": 1.0}
        # Don't know what this is. Value taken from ComfyUI Mochi model.
        self.memory_usage_factor = 2.0
        # denoiser is handled by extension
        self.unet_config["disable_unet_model_creation"] = True

def load_video_frames(video_path):
    image_files = sorted(glob.glob(os.path.join(video_path, "*.png")))  # Assuming PNG images
    if not image_files:
        raise ValueError(f"No image files found in the specified folder: {video_path}")
    frames = []
    for image_file in image_files:
        try:
            img = Image.open(image_file).convert("RGB")
            frames.append(img)
        except Exception as e:
            print(f"Warning: Could not load image {image_file}. Skipping. Error: {e}")
    if not frames:
        raise ValueError("No valid images were loaded.")
    return frames

def save_video_ffmpeg(frames, output_path, fps=24):
    """
    Save a sequence of frames as a video file using imageio_ffmpeg directly.
    """
    try:
        first_frame = frames[0]
        if hasattr(first_frame, 'numpy'):
            first_frame = first_frame.numpy()
        elif hasattr(first_frame, 'getdata'):
            first_frame = np.array(first_frame)
            
        height, width = first_frame.shape[:2]
        
        # Initialize writer
        writer = imageio_ffmpeg.write_frames(
            output_path,
            size=(width, height),
            fps=fps,
            codec='libx264',
            output_params=[
                '-preset', 'medium',
                '-crf', '17',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart'
            ]
        )
        
        # Start the writer
        writer.send(None)  # Initialize the generator
        
        # Write frames
        try:
            for frame in frames:
                if hasattr(frame, 'numpy'):
                    frame = frame.numpy()
                elif hasattr(frame, 'getdata'):
                    frame = np.array(frame)
                    
                writer.send(frame)
        finally:
            writer.close()
        
        print(f"Video saved successfully to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error saving video: {str(e)}")
        import traceback
        traceback.pr

def resize_frames(frames, width, height):
    resized_frames = []
    for frame in frames:
        resized_frame = T.functional.resize(frame, (height, width), interpolation=T.InterpolationMode.LANCZOS)
        resized_frames.append(resized_frame)
    return resized_frames

def get_rotary_pos_embed_hyvideo(transformer, video_length, height, width):
    # Simplified from the original get_rotary_pos_embed
    target_ndim = 3
    rope_theta = 225
    patch_size = transformer.patch_size
    rope_dim_list = transformer.rope_dim_list
    hidden_size = transformer.hidden_size
    heads_num = transformer.heads_num
    head_dim = hidden_size // heads_num

    latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]

    if isinstance(patch_size, int):
        rope_sizes = [s // patch_size for s in latents_size]
    elif isinstance(patch_size, list):
        rope_sizes = [s // patch_size[idx] for idx, s in enumerate(latents_size)]
    else:
        raise ValueError("patch_size must be int or list")

    if len(rope_sizes) != target_ndim:
        rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes

    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]

    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        rope_dim_list,
        rope_sizes,
        theta=rope_theta,
        use_real=True,
        theta_rescale_factor=1,
    )
    return freqs_cos, freqs_sin

def load_text_encoder(precision, device, offload_device, apply_final_norm=False, hidden_state_skip_layer=2, quantization="disabled"):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    if quantization == "bnb_nf4":
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        quantization_config = None

    # Replace with actual loading logic for CLIP
    text_encoder_2 = TextEncoder(
        text_encoder_path=CLIP_PATH,
        text_encoder_type="clipL",
        max_length=77,
        text_encoder_precision=precision,
        tokenizer_type="clipL",
        logger=None,
        device=device,
    )

    # Replace with actual loading logic for LLM
    text_encoder = TextEncoder(
        text_encoder_path=LLM_PATH,
        text_encoder_type="llm",
        max_length=256,
        text_encoder_precision=precision,
        tokenizer_type="llm",
        hidden_state_skip_layer=hidden_state_skip_layer,
        apply_final_norm=apply_final_norm,
        logger=None,
        device=device,
        dtype=dtype,
        quantization_config=quantization_config
    )

    return text_encoder, text_encoder_2

def encode_text(text_encoder, text_encoder_2, prompt, device, offload_device, force_offload, prompt_template="video", custom_prompt_template=None):
    negative_prompt = None

    if prompt_template != "disabled":
        if prompt_template == "custom":
            prompt_template_dict = custom_prompt_template
        elif prompt_template == "video":
            prompt_template_dict = PROMPT_TEMPLATE["dit-llm-encode-video"]
        elif prompt_template == "image":
            prompt_template_dict = PROMPT_TEMPLATE["dit-llm-encode"]
        else:
            raise ValueError(f"Invalid prompt_template: {prompt_template_dict}")
        assert (
            isinstance(prompt_template_dict, dict)
            and "template" in prompt_template_dict
        ), f"`prompt_template` must be a dictionary with a key 'template', got {prompt_template_dict}"
        assert "{}" in str(prompt_template_dict["template"]), (
            "`prompt_template['template']` must contain a placeholder `{}` for the input text, "
            f"got {prompt_template_dict['template']}"
        )
    else:
        prompt_template_dict = None

    def encode_single_prompt(text_encoder, prompt, negative_prompt, device):
        text_inputs = text_encoder.text2tokens(prompt, prompt_template=prompt_template_dict)
        prompt_outputs = text_encoder.encode(text_inputs, prompt_template=prompt_template_dict, device=device)
        prompt_embeds = prompt_outputs.hidden_state
        attention_mask = prompt_outputs.attention_mask

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            attention_mask = attention_mask.view(1, -1)

        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
        negative_prompt_embeds = None
        negative_attention_mask = None

        return prompt_embeds, negative_prompt_embeds, attention_mask, negative_attention_mask

    text_encoder.to(device)
    prompt_embeds, negative_prompt_embeds, attention_mask, negative_attention_mask = encode_single_prompt(text_encoder, prompt, negative_prompt, device)
    if force_offload:
        text_encoder.to(offload_device)

    if text_encoder_2 is not None:
        text_encoder_2.to(device)
        prompt_embeds_2, negative_prompt_embeds_2, attention_mask_2, negative_attention_mask_2 = encode_single_prompt(text_encoder_2, prompt, negative_prompt, device)
        if force_offload:
            text_encoder_2.to(offload_device)
    else:
        prompt_embeds_2, negative_prompt_embeds_2, attention_mask_2, negative_attention_mask_2 = None, None, None, None

    return {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "attention_mask": attention_mask,
        "negative_attention_mask": negative_attention_mask,
        "prompt_embeds_2": prompt_embeds_2,
        "negative_prompt_embeds_2": negative_prompt_embeds_2,
        "attention_mask_2": attention_mask_2,
        "negative_attention_mask_2": negative_attention_mask_2,
    }

def load_model(model_path, device, offload_device, base_dtype, quant_dtype, compile_args=None, attention_mode="sdpa"):
    # 1. Define channels and factor_kwargs
    in_channels = out_channels = 16
    factor_kwargs = {"device": device, "dtype": base_dtype}

    print(f'load_model {model_path}, {device}, {offload_device}, {quant_dtype}, {base_dtype}')
    
    # 2. Initialize an empty model
    HUNYUAN_VIDEO_CONFIG = {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    }
    
    with init_empty_weights():
        transformer = HYVideoDiffusionTransformer(
            in_channels=in_channels,
            out_channels=out_channels,
            attention_mode=attention_mode,
            main_device=device,
            offload_device=offload_device,
            **HUNYUAN_VIDEO_CONFIG,
            **factor_kwargs
        )
    transformer.eval()

    # 3. Load the state dict
    sd = load_torch_file(model_path, device=offload_device)

    # 4. Create a ModelPatcher
    comfy_model = HyVideoModel(
        HyVideoModelConfig(base_dtype),
        model_type=ModelType.FLOW,
        device=device,
    )

    print("Using accelerate to load and assign model weights to device...")

    dtype = torch.float8_e4m3fn
    params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
    for name, param in transformer.named_parameters():
        dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
        set_module_tensor_to_device(transformer, name, device=device, dtype=dtype_to_use, value=sd[name])

    comfy_model.diffusion_model = transformer
    patcher = ModelPatcher(comfy_model, device, offload_device)

    del sd
    gc.collect()
    mm.soft_empty_cache()
    mm.load_model_gpu(patcher)

    patcher.model.diffusion_model.to(offload_device)

    # 6. Create HunyuanVideoPipeline instance
    pipe = HunyuanVideoPipeline(
        transformer=transformer,
        scheduler=FlowMatchDiscreteScheduler(
            shift=9.0,
            reverse=True,
            solver="euler",
        ),
        progress_bar_config=None,
        base_dtype=base_dtype
    )

    patcher.model["pipe"] = pipe
    patcher.model["dtype"] = base_dtype
    patcher.model["base_path"] = model_path
    patcher.model["model_name"] = transformer
    patcher.model["manual_offloading"] = False
    patcher.model["quantization"] = "disabled"
    patcher.model["block_swap_args"] = None

    return patcher, pipe

def load_vae(vae_path, device, offload_device, precision, compile_args=None):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

    # Load VAE configuration
    import json
    with open(VAE_CONFIG, 'r') as f:
        vae_config = json.load(f)

    # Load VAE state dict
    vae_sd = load_torch_file(vae_path, device=offload_device)

    # Initialize and load VAE
    vae = AutoencoderKLCausal3D.from_config(vae_config)
    vae.load_state_dict(vae_sd)
    del vae_sd
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device=device, dtype=dtype)

    # # Compile VAE if requested
    # if compile_args:
    #     torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
    #     vae = torch.compile(vae, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

    return vae

def encode_video(vae, frames, device, offload_device):
    # Assuming frames are PIL Images
    vae.to(device)
    vae.enable_tiling()
    vae.tile_latent_min_tsize = 16
    vae.tile_sample_min_size = 256
    vae.tile_latent_min_size = 32

    # Convert PIL Images to tensors and normalize
    tensor_frames = []
    for frame in frames:
        frame = T.functional.pil_to_tensor(frame)
        tensor_frames.append(frame)

    # Stack frames into a single tensor
    tensor_frames = torch.stack(tensor_frames, dim=0).unsqueeze(0)  # Shape: [1, T, C, H, W]
    tensor_frames = tensor_frames.permute(0, 2, 1, 3, 4).float()  # Shape: [1, C, T, H, W]
    tensor_frames = (tensor_frames / 255.0) * 2.0 - 1.0
    tensor_frames = tensor_frames.to(vae.dtype).to(device)

    # Encode
    generator = torch.Generator(device=torch.device("cpu"))
    latents = vae.encode(tensor_frames).latent_dist.sample(generator)
    latents = latents * vae.config.scaling_factor

    vae.to(offload_device)

    return latents

def sample_video(pipeline, text_embeddings, latents, width, height, num_frames, steps, embedded_guidance_scale, flow_shift, seed, device, offload_device, denoise_strength=1.0, force_offload=True):
    print(f'sample_video {width}, {height}, {steps}, {embedded_guidance_scale}, {flow_shift}, {seed}')

    dtype = BASE_DTYPE
    transformer = pipeline.transformer

    # stg_args = {
    #     "stg_mode": "STG-A", #(["STG-A", "STG-R"],),
    #     "stg_block_idx": 0, #("INT", {"default": 0, "min": -1, "max": 39, "step": 1, "tooltip": "Block index to apply STG"}),
    #     "stg_scale": 1.0, #("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Recommended values are ≤2.0"}),
    #     "stg_start_percent": 0.0, #("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the steps to apply STG"}),
    #     "stg_end_percent": 1.0, #("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the steps to apply STG"}),
    # }

    # #handle STG
    # if stg_args is not None:
    #     if stg_args["stg_mode"] == "STG-A" and transformer.attention_mode != "sdpa":
    #         raise ValueError(
    #             f"STG-A requires attention_mode to be 'sdpa', but got {transformer.attention_mode}."
    #     )
    #handle CFG
    if text_embeddings.get("cfg") is not None:
        cfg = float(text_embeddings.get("cfg", 1.0))
        cfg_start_percent = float(text_embeddings.get("start_percent", 0.0))
        cfg_end_percent = float(text_embeddings.get("end_percent", 1.0))
    else:
        cfg = 1.0
        cfg_start_percent = 0.0
        cfg_end_percent = 1.0

    generator = torch.Generator(device=torch.device("cpu")).manual_seed(seed)

    target_height = align_to(height, 16)
    target_width = align_to(width, 16)

    freqs_cos, freqs_sin = get_rotary_pos_embed_hyvideo(
        pipeline.transformer, num_frames, target_height, target_width
    )
    n_tokens = freqs_cos.shape[0]

    pipeline.scheduler.shift = flow_shift

    if force_offload:
        print(f'force offload')
        transformer.to(device)

    mm.soft_empty_cache()
    gc.collect()

    out_latents = pipeline(
        num_inference_steps=steps,
        height=target_height,
        width=target_width,
        video_length=num_frames,
        guidance_scale=cfg,
        cfg_start_percent=cfg_start_percent,
        cfg_end_percent=cfg_end_percent,
        embedded_guidance_scale=embedded_guidance_scale,
        latents=latents,
        denoise_strength=denoise_strength,
        prompt_embed_dict=text_embeddings,
        generator=generator,
        freqs_cis=(freqs_cos, freqs_sin),
        n_tokens=n_tokens,
        # stg_mode=stg_args["stg_mode"] if stg_args is not None else None,
        # stg_block_idx=stg_args["stg_block_idx"] if stg_args is not None else -1,
        # stg_scale=stg_args["stg_scale"] if stg_args is not None else 0.0,
        # stg_start_percent=stg_args["stg_start_percent"] if stg_args is not None else 0.0,
        # stg_end_percent=stg_args["stg_end_percent"] if stg_args is not None else 1.0,
    )

    if force_offload:
        transformer.to(offload_device)
        mm.soft_empty_cache()
        gc.collect()

    return out_latents

def decode_video(vae, latents, device, offload_device):
    try:
        vae.to(device)
        vae.enable_tiling()
        vae.tile_latent_min_tsize = 16
        vae.tile_sample_min_size = 256
        vae.tile_latent_min_size = 32

        # # Handle input dimensions
        # if len(latents.shape) == 4:
        #     if isinstance(vae, AutoencoderKLCausal3D):
        #         latents = latents.unsqueeze(2)
        # elif len(latents.shape) != 5:
        #     raise ValueError(
        #         f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
        #     )

        print(f'Unprocessed shape: {latents.shape}')

        latents = latents / vae.config.scaling_factor
        latents = latents.to(vae.dtype).to(device)

        # Create CPU generator for reproducibility
        generator = torch.Generator(device=torch.device("cpu"))

        # Decode latents
        with torch.no_grad():
            video = vae.decode(latents, return_dict=False, generator=generator)[0]

        print(f'Raw decoded shape: {latents.shape}')

        video_processor = VideoProcessor(vae_scale_factor=8)
        video_processor.config.do_resize = False
        video = video_processor.postprocess_video(video=video, output_type="pt")

        frames = video.squeeze(0)
        frames = frames.permute(0, 2, 3, 1).cpu().float()  # [F, H, W, C]

        # Convert tensors to PIL Images
        pil_frames = []
        # os.makedirs("debug_frames", exist_ok=True)
        for idx, frame in enumerate(frames):
            frame_uint8 = (frame * 255).round().to(torch.uint8).numpy()
            pil_frame = Image.fromarray(frame_uint8, mode='RGB')
            # pil_frame.save(f"debug_frames/frame_{idx:04d}.png")
            pil_frames.append(pil_frame)

        vae.to(offload_device)

        return pil_frames
    except Exception as e:
        print(f"Error in decode_video: {str(e)}")
        raise

# Main execution
if __name__ == "__main__":
    # 1. Define devices for each  model in the pipeline
    # and then load them passing the device itself as the offload device
    # effectively disabling offloading.
    model_device = torch.device("cuda:0")
    vae_device = torch.device("cuda:1")
    text_encoder_device = torch.device("cuda:2")
    sample_device = torch.device("cuda:3")
    
    transformer, pipeline = load_model(MODEL_PATH, model_device, model_device, BASE_DTYPE, QUANT_DTYPE, compile_args=None, attention_mode="sdpa")
    vae = load_vae(VAE_PATH, vae_device, vae_device, "bf16", compile_args=None)
    text_encoder, text_encoder_2 = load_text_encoder("fp16", text_encoder_device, text_encoder_device)

    # 2. Load and preprocess input video
    frames = load_video_frames(INPUT_FRAMES_PATH)
    frames = resize_frames(frames, WIDTH, HEIGHT)
    print(f"Loaded and resized {len(frames)} frames.")

    # 3. Encode input video
    latents = encode_video(vae, frames, vae_device, vae_device)
    print("Encoded video into latents.")

    # 4. Encode prompt
    text_embeddings = encode_text(text_encoder, text_encoder_2, PROMPT, text_encoder_device, text_encoder_device, FORCE_OFFLOAD, prompt_template="video")
    print("Encoded text prompt.")

    # # Test
    # new_frames = decode_video(vae, latents, vae_device, vae_device)
    # print("Decoded latents into frames. (test mode)")

    # 5. Sample
    new_latents = sample_video(pipeline, text_embeddings, latents, WIDTH, HEIGHT, NUM_FRAMES, STEPS, EMBEDDED_GUIDANCE_SCALE, FLOW_SHIFT, SEED, sample_device, sample_device, DENOISE_STRENGTH)
    print("Sampled new latents.")

    # 6. Decode
    new_frames = decode_video(vae, new_latents, vae_device, vae_device)
    print("Decoded latents into frames.")

    # 7. Combine frames and save
    save_video_ffmpeg(new_frames, OUTPUT_VIDEO, fps=24)
