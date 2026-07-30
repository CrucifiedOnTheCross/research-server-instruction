from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as vision

from tools.stage16g_training_state import completed_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 16G-P1 dermoscopy domain LoRA.")
    parser.add_argument("--config", default="configs/stage16g_p1_domain_lora.yaml")
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


class BalancedDomainDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path,
        resolution: int,
        seed: int,
        length: int,
        horizontal_flip: bool,
        start_index: int = 0,
    ):
        self.data_root = data_root
        self.resolution = resolution
        self.seed = seed
        self.length = length
        self.horizontal_flip = horizontal_flip
        self.start_index = start_index
        self.labels = sorted(frame["label"].unique())
        self.by_label = {
            label: frame[frame["label"] == label].sort_values("group_id").to_dict("records")
            for label in self.labels
        }

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        index += self.start_index
        label = self.labels[index % len(self.labels)]
        rows = self.by_label[label]
        cycle = index // len(self.labels)
        row = rows[(cycle * 15485863 + self.seed) % len(rows)]
        image = Image.open(self.data_root / str(row["image_path"])).convert("RGB")
        image = ImageOps.fit(
            image,
            (self.resolution, self.resolution),
            method=Image.Resampling.LANCZOS,
        )
        if self.horizontal_flip and ((index + self.seed) % 2 == 0):
            image = ImageOps.mirror(image)
        tensor = vision.to_tensor(image) * 2.0 - 1.0
        return {"pixel_values": tensor, "caption": str(row["caption"])}


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latest_training_state(output_root: Path) -> Path | None:
    states = sorted(
        output_root.glob("checkpoint-*/training_state.pt"),
        key=lambda path: int(path.parent.name.split("-")[-1]),
    )
    return states[-1] if states else None


def main() -> None:
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import compute_snr
    from diffusers.utils import convert_state_dict_to_diffusers
    from huggingface_hub import model_info
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    training = config["training"]
    max_steps = int(args.max_train_steps or training["max_train_steps"])
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 16G-P1 requires CUDA")

    output_root = Path(args.output_root or config["data"]["output_root"])
    lora_root = output_root / "lora"
    output_root.mkdir(parents=True, exist_ok=True)
    existing_summary = completed_summary(output_root, max_steps)
    if existing_summary is not None:
        print(json.dumps(existing_summary, indent=2))
        return
    frame = pd.read_csv(config["data"]["training_manifest"])
    data_root = Path(config["data"]["root"])
    batch_size = int(training["train_batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    resume_path = latest_training_state(output_root)
    resume_state = (
        torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_path
        else None
    )
    initial_step = int(resume_state["global_step"]) if resume_state else 0
    remaining_micro_batches = (max_steps - initial_step) * accumulation
    dataset = BalancedDomainDataset(
        frame,
        data_root,
        int(training["resolution"]),
        seed,
        remaining_micro_batches * batch_size,
        bool(training["horizontal_flip"]),
        start_index=initial_step * batch_size * accumulation,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    model_id = config["model"]["base_model_id"]
    revision = str(model_info(model_id).sha)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", revision=revision)
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", revision=revision, torch_dtype=torch.bfloat16
    ).cuda().eval()
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", revision=revision, torch_dtype=torch.bfloat16
    ).cuda().eval()
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", revision=revision, torch_dtype=torch.bfloat16
    ).cuda()
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_id, subfolder="scheduler", revision=revision
    )
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    unet.add_adapter(
        LoraConfig(
            r=int(config["model"]["rank"]),
            lora_alpha=int(config["model"]["lora_alpha"]),
            init_lora_weights="gaussian",
            target_modules=list(config["model"]["target_modules"]),
        )
    )
    for parameter in unet.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    if training["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    trainable = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = get_scheduler(
        str(training["lr_scheduler"]),
        optimizer,
        num_warmup_steps=int(training["lr_warmup_steps"]),
        num_training_steps=max_steps,
    )
    if resume_state:
        from peft.utils import set_peft_model_state_dict

        set_peft_model_state_dict(unet, resume_state["lora_state"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])

    prompts = list(config["prompts"].values())
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prompt_embeddings = text_encoder(tokenized, return_dict=False)[0]
    prompt_map = {
        prompt: prompt_embeddings[index : index + 1].detach()
        for index, prompt in enumerate(prompts)
    }
    del text_encoder
    torch.cuda.empty_cache()

    environment = {
        "protocol": config["protocol"],
        "test_evaluated": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "base_model_id": model_id,
        "base_model_revision": revision,
        "peft_trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        "max_train_steps": max_steps,
        "resumed_from_step": initial_step,
        "effective_batch_size": batch_size * accumulation,
    }
    (output_root / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    metrics_path = output_root / "metrics.csv"
    metrics: list[dict[str, Any]] = (
        pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    )
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    global_step = initial_step
    micro_step = 0
    unet.train()
    for batch in loader:
        micro_step += 1
        images = batch["pixel_values"].to(
            device="cuda", dtype=torch.bfloat16, non_blocking=True
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device="cuda"
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        embeddings = torch.cat([prompt_map[prompt] for prompt in batch["caption"]])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = unet(noisy_latents, timesteps, embeddings, return_dict=False)[0]
        target = (
            noise
            if noise_scheduler.config.prediction_type == "epsilon"
            else noise_scheduler.get_velocity(latents, noise, timesteps)
        )
        snr = compute_snr(noise_scheduler, timesteps)
        weights = torch.stack(
            [snr, float(training["snr_gamma"]) * torch.ones_like(snr)], dim=1
        ).min(dim=1)[0]
        if noise_scheduler.config.prediction_type == "epsilon":
            weights = weights / snr
        else:
            weights = weights / (snr + 1)
        loss = functional.mse_loss(
            prediction.float(), target.float(), reduction="none"
        ).mean(dim=(1, 2, 3))
        loss = (loss * weights).mean()
        (loss / accumulation).backward()
        if micro_step % accumulation:
            continue
        torch.nn.utils.clip_grad_norm_(trainable, float(training["max_grad_norm"]))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        if global_step == 1 or global_step % int(training["metrics_every_steps"]) == 0:
            metrics.append(
                {
                    "step": global_step,
                    "loss": float(loss.detach()),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "elapsed_seconds": time.time() - started,
                    "gpu_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "gpu_memory_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                }
            )
            write_metrics(metrics_path, metrics)
        if global_step % int(training["checkpointing_steps"]) == 0 or global_step == max_steps:
            state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
            checkpoint_root = output_root / f"checkpoint-{global_step}"
            StableDiffusionPipeline.save_lora_weights(
                checkpoint_root, unet_lora_layers=state, safe_serialization=True
            )
            torch.save(
                {
                    "global_step": global_step,
                    "lora_state": {
                        key: value.detach().float().cpu()
                        for key, value in get_peft_model_state_dict(unet).items()
                    },
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                checkpoint_root / "training_state.pt",
            )
        if global_step >= max_steps:
            break

    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(
        lora_root, unet_lora_layers=state, safe_serialization=True
    )
    summary = {
        **environment,
        "completed_steps": global_step,
        "complete": global_step == max_steps,
        "final_loss": float(metrics[-1]["loss"]),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "lora_path": str(lora_root),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
