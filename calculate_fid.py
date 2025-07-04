import argparse
import logging
import os
from pathlib import Path
import json
import ast

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from utils import load_encoders

from dataset import CustomDataset
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import ConvertImageDtype

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 


def free_gpu_memory(*tensors):
    """Delete provided tensors (if any) and empty the CUDA cache to proactively
    release VRAM. Use this **sparingly** inside tight loops when you are sure
    the tensors are no longer needed."""
    for t in tensors:
        try:
            del t
        except Exception:
            # If the tensor is already out of scope or None, ignore.
            pass


@torch.no_grad()
def calculate_fid(model, vae, val_dataloader, accelerator, args, num_samples=50000, 
                 latents_scale=None, latents_bias=None, num_steps=50, cfg_scale=1.0, 
                 guidance_low=1.0, guidance_high=1.0):
    """
    Calculate FID score using generated and real images from validation set across all GPUs.
    """
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if latents_scale is not None:
        latents_scale = latents_scale.to(accelerator.device, dtype=dtype)
    if latents_bias is not None:
        latents_bias = latents_bias.to(accelerator.device, dtype=dtype)

    model.eval()
    
    # Initialize FID metric on each GPU
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(accelerator.device)
    rescaler_to_uint8 = ConvertImageDtype(torch.uint8)
    rescaler_to_0_1 = lambda x: (x + 1.0) / 2.0  # Convert [-1, 1] to [0, 1]
    
    # Calculate samples per process
    samples_per_process = num_samples // accelerator.num_processes
    
    # Move VAE to device
    vae = vae.to(accelerator.device)
    
    samples_processed = 0

    expected_batches = samples_per_process // args.batch_size
    
    for batch_idx, (raw_image, x, y) in enumerate(tqdm(val_dataloader, desc=f"Calculating FID (steps={num_steps}, cfg={cfg_scale}, g_low={guidance_low}, g_high={guidance_high})", disable=not accelerator.is_main_process, total=expected_batches)):
        if samples_processed >= samples_per_process:
            break
            
        raw_image = raw_image.to(accelerator.device, non_blocking=True)
        x = x.squeeze(dim=1).to(accelerator.device, non_blocking=True)
        y = y.to(accelerator.device, non_blocking=True)
        
        batch_size = x.shape[0]
        
        with torch.no_grad():
            # Create noise on the correct device
            noise = torch.randn(x.shape[0], 4, x.shape[2], x.shape[3], device=accelerator.device, dtype=dtype)
            from samplers import euler_maruyama_sampler
            with accelerator.autocast():
                fake_latents = euler_maruyama_sampler(
                    model, 
                    noise, 
                    y,
                    num_steps=num_steps, 
                    cfg_scale=cfg_scale,
                    guidance_low=guidance_low,
                    guidance_high=guidance_high,
                    path_type=args.path_type,
                    heun=False,
                )
            
                fake_images = vae.decode(((fake_latents - latents_bias) / latents_scale).to(dtype=dtype)).sample.clamp(-1, 1)

        # Convert to uint8 for FID calculation
        fake_images_uint8 = rescaler_to_uint8(rescaler_to_0_1(fake_images))
        
        # Update FID
        fid.update(raw_image, real=True)
        fid.update(fake_images_uint8, real=False)

        free_gpu_memory(
            raw_image,
            x,
            y,
            fake_latents,
            fake_images,
            fake_images_uint8,
        )

        samples_processed += batch_size
    
    # Synchronize across all processes
    accelerator.wait_for_everyone()
    
    # Compute FID score
    fid_score = fid.compute()
    fid_value = fid_score.item() if torch.is_tensor(fid_score) else fid_score
    
    # Cleanup
    del fid
    vae = vae.to("cpu")
    torch.cuda.empty_cache()
    
    return fid_value


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/fid_log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def main(args):    
    # set accelerator
    logging_dir = str(Path(args.output_dir, "fid_logs"))
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        logger = create_logger(logging_dir)
        logger.info(f"Starting FID calculation")

    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    if args.enc_type != None:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
            )
    else:
        raise NotImplementedError()
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs
    )

    model = model.to(device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load checkpoint
    if accelerator.is_main_process:
        logger.info(f"Loading checkpoint from {args.checkpoint_path}")
    
    ckpt = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['ema'])  # Use EMA weights
    
    # Setup data:
    val_dataset = CustomDataset(os.path.join(args.data_dir, "validation"))
    
    # Create distributed sampler for validation dataset to ensure even distribution across GPUs
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,  # Keep deterministic order for consistent validation
        drop_last=False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,  # Use distributed sampler instead of shuffle
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    if accelerator.is_main_process:
        logger.info(f"Validation dataset contains {len(val_dataset):,} images ({os.path.join(args.data_dir, 'validation')})")
    
    model.eval()


    # Parse parameter lists
    steps_list = args.steps if isinstance(args.steps, list) else [args.steps]
    cfg_scale_list = args.cfg_scale if isinstance(args.cfg_scale, list) else [args.cfg_scale]
    guidance_low_list = args.guidance_low if isinstance(args.guidance_low, list) else [args.guidance_low]
    guidance_high_list = args.guidance_high if isinstance(args.guidance_high, list) else [args.guidance_high]
    
    results = {}
    
    # Run ablation over all combinations
    for num_steps in steps_list:
        for cfg_scale in cfg_scale_list:
            for guidance_low in guidance_low_list:
                for guidance_high in guidance_high_list:
                    if accelerator.is_main_process:
                        logger.info(f"Calculating FID with steps={num_steps}, cfg_scale={cfg_scale}, guidance_low={guidance_low}, guidance_high={guidance_high}")
                    
                    fid_score = calculate_fid(
                        model, vae, val_dataloader,
                        accelerator, args, 
                        num_samples=args.num_samples, 
                        latents_scale=latents_scale, 
                        latents_bias=latents_bias,
                        num_steps=num_steps,
                        cfg_scale=cfg_scale,
                        guidance_low=guidance_low,
                        guidance_high=guidance_high
                    )
                    
                    if accelerator.is_main_process:
                        logger.info(f"FID Score (steps={num_steps}, cfg={cfg_scale}, g_low={guidance_low}, g_high={guidance_high}): {fid_score:.4f}")
                        results[f"steps_{num_steps}_cfg_{cfg_scale}_glow_{guidance_low}_ghigh_{guidance_high}"] = fid_score
                

    
    # Save results
    if accelerator.is_main_process:
        results_file = os.path.join(args.output_dir, "fid_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {results_file}")
        
        # Print summary
        logger.info("=== FID Results Summary ===")
        for key, value in results.items():
            logger.info(f"{key}: {value:.4f}")
        
        # Find best configuration
        best_config = min(results.items(), key=lambda x: x[1])
        logger.info(f"Best configuration: {best_config[0]} with FID: {best_config[1]:.4f}")


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("FID calculation completed!")


def parse_list_arg(value):
    """Parse command line argument that can be either a single value or a list"""
    try:
        # Try to parse as a list
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        else:
            return [parsed]
    except (ValueError, SyntaxError):
        # If parsing fails, treat as single value
        try:
            return [float(value)]
        except ValueError:
            return [int(value)]


def parse_args():
    parser = argparse.ArgumentParser(description="FID Calculation for REPA")

    # Model and checkpoint
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default="SiT-XL/2", help="Model architecture")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # Dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256", help="Path to validation data")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for FID calculation")

    # Sampling parameters (can be lists for ablation)
    parser.add_argument("--steps", type=str, default="50", 
                        help="Number of sampling steps. Can be single value or list like '[25, 50, 100]'")
    parser.add_argument("--cfg-scale", type=str, default="1.0", 
                        help="CFG scale values. Can be single value or list like '[1.0, 1.5, 2.0]'")
    parser.add_argument("--guidance-low", type=str, default="1.0", 
                        help="Guidance low values. Can be single value or list like '[1.0, 1.5, 2.0]'")
    parser.add_argument("--guidance-high", type=str, default="1.0", 
                        help="Guidance high values. Can be single value or list like '[1.0, 1.5, 2.0]'")

    # FID parameters
    parser.add_argument("--num-samples", type=int, default=50000, help="Number of samples for FID calculation")

    # Technical parameters
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Loss parameters (needed for model loading)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')

    # Logging
    parser.add_argument("--output-dir", type=str, default="fid_results", help="Output directory for results")

    args = parser.parse_args()
    
    # Parse list arguments
    args.steps = parse_list_arg(args.steps)
    args.cfg_scale = parse_list_arg(args.cfg_scale)
    args.guidance_low = parse_list_arg(args.guidance_low)
    args.guidance_high = parse_list_arg(args.guidance_high)
    
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args) 