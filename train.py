import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from loss import SILoss
from utils import load_encoders

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import ConvertImageDtype
import torchvision

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
def calculate_fid(model, vae, val_dataloader, accelerator, args, num_samples=50000, latents_scale=None, latents_bias=None):
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
    
    for raw_image, x, y in tqdm(val_dataloader, desc="Calculating FID", disable=not accelerator.is_main_process, total=samples_per_process//(args.batch_size//accelerator.num_processes)):
        if samples_processed >= samples_per_process:
            break
            
        raw_image = raw_image.to(accelerator.device, non_blocking=True)
        x = x.squeeze(dim=1).to(accelerator.device, non_blocking=True)
        y = y.to(accelerator.device, non_blocking=True)
        
        batch_size = x.shape[0]
        
        with torch.no_grad():
            # Create noise on the correct device
            noise = torch.randn(x.shape[0], 4, x.shape[2], x.shape[3], device=accelerator.device, dtype=dtype)
            from samplers import euler_sampler
            with accelerator.autocast():
                fake_latents = euler_sampler(
                    model, 
                    noise, 
                    y,
                    num_steps=50, 
                    cfg_scale=1.0,
                    guidance_low=0.,
                    guidance_high=1.,
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
    
    model.train()
    return fid_value


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
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
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)
    
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data:
    train_dataset = CustomDataset(os.path.join(args.data_dir, "train"))
    val_dataset = CustomDataset(os.path.join(args.data_dir, "validation"))
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
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
        batch_size=local_batch_size,
        sampler=val_sampler,  # Use distributed sampler instead of shuffle
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    if accelerator.is_main_process:
        logger.info(f"Train dataset contains {len(train_dataset):,} images ({os.path.join(args.data_dir, 'train')})")
        logger.info(f"Validation dataset contains {len(val_dataset):,} images ({os.path.join(args.data_dir, 'validation')})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            weights_only=False,
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REPA", 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    gt_raw_images, gt_xs, _ = next(iter(train_dataloader))
    assert gt_raw_images.shape[-1] == args.resolution
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
        )
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)
        
    for epoch in range(args.epochs):
        model.train()
        for raw_image, x, y in train_dataloader:
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            z = None
            if args.legacy:
                # In our early experiments, we accidentally apply label dropping twice: 
                # once in train.py and once in sit.py. 
                # We keep this option for exact reproducibility with previous runs.
                drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, y)
            else:
                labels = y
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                zs = []
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                        zs.append(z)

            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                loss, proj_loss, contrastive_flow_loss = loss_fn(model, x, model_kwargs, zs=zs)
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()
                contrastive_flow_loss_mean = contrastive_flow_loss.mean()
                loss = (loss_mean - contrastive_flow_loss_mean * args.contrastive_flow_coeff) + proj_loss_mean * args.proj_coeff
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict() if accelerator.num_processes > 1 else model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                # Calculate FID
                if args.enable_fid:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        logger.info(f"Calculating FID at step {global_step}")
                    
                    with accelerator.autocast():
                        fid_score = calculate_fid(
                            ema, vae, val_dataloader,
                            accelerator, args, num_samples=args.fid_samples, 
                            latents_scale=latents_scale, latents_bias=latents_bias
                        )
                    
                    if accelerator.is_main_process:
                        logger.info(f"FID Score: {fid_score:.4f}")
                        # Log FID to wandb
                        accelerator.log({"fid_score": fid_score}, step=global_step)

            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                vae = vae.to(device)
                from samplers import euler_sampler
                with torch.no_grad():
                    samples = euler_sampler(
                        ema, 
                        xT, 
                        ys,
                        num_steps=50, 
                        cfg_scale=1.85,
                        guidance_low=0.,
                        guidance_high=0.65,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)
                    samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                    gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                
                vae = vae.to("cpu")

                accelerator.log({"samples": wandb.Image(array2grid(out_samples)),
                                    "gt_samples": wandb.Image(array2grid(gt_samples))}, step=global_step)
                logging.info("Generating EMA samples done.")

            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(), 
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                "contrastive_flow_loss": accelerator.gather(contrastive_flow_loss_mean).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--contrastive-flow-coeff", type=float, default=0.05)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    # Evaluation parameters
    parser.add_argument("--enable-fid", action="store_true", default=True, help="Enable FID calculation during checkpointing")
    parser.add_argument("--fid-samples", type=int, default=50000, help="Number of samples for FID calculation")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
