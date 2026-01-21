from collections import OrderedDict
import logging
import os
from glob import glob
from copy import deepcopy
from time import time
from typing import Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from rich.progress import track
from widit import WiDiT
try:
    import wandb  # optional
except Exception:
    wandb = None

from .diffusion import create_diffusion


def create_logger(logging_dir: str):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    # Prevent duplicate handlers
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(f"{logging_dir}/log.txt")
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger


def get_state_dict_for_saving(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def _run_validation_loop(
    accelerator: Accelerator,
    model_for_eval: torch.nn.Module,
    diffusion,
    dataloader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    use_diffusion: bool,
    criterion = torch.nn.SmoothL1Loss(reduction="mean"),
    timestep_seed: int = 42,
) -> float:
    model_for_eval.eval()
    total_loss, total_batches = 0.0, 0
    rng = torch.Generator(device=device).manual_seed(timestep_seed)

    for batch in track(dataloader, total=len(dataloader), description="Validation:"):
        if not isinstance(batch, (list, tuple)) or len(batch) not in {2, 3}:
            raise ValueError("Validation dataloader must return (x, target) or (x, target, timestep).")
        x, target = batch[0], batch[1]
        timestep = batch[2] - 1 if len(batch) == 3 else None

        x = x.to(device=device, dtype=dtype, non_blocking=True)
        target = target.to(device=device, dtype=dtype, non_blocking=True)

        if use_diffusion:
            # Mirror training: model(input=target, conditioned=x)
            t = torch.randint(
                0,
                diffusion.num_timesteps,
                (x.shape[0],),
                device=device,
                generator=rng,
            )
            noise = torch.randn(
                target.shape,
                dtype=target.dtype,
                device=target.device,
                generator=rng,
            )
            loss_dict = diffusion.training_losses(
                model_for_eval,
                target,
                t,
                dict(conditioned=x),
                noise=noise,
            )
            loss = loss_dict["mse"].mean()
        else:
            y = model_for_eval(x, timestep=timestep)
            loss = criterion(y, target)

        loss = accelerator.reduce(loss, reduction="mean")
        total_loss += loss.item()
        total_batches += 1

    return (total_loss / max(total_batches, 1)) if total_batches > 0 else float("nan")


def train(
    model: WiDiT,
    training_dataloader: DataLoader,
    validation_dataloader: DataLoader | None = None,
    results_dir: str = "results",
    epochs: int = 40,
    log_every: int = 10,
    use_diffusion: bool = True,
    criterion = torch.nn.MSELoss(reduction="mean"),
    precision: str = "fp16",
    wandb_logging: bool = False,
    wandb_project: str = "WiDiT",
    run_name: str|None = None,
    wandb_config: Optional[Dict] = None,
    wandb_log_artifacts: bool = False,
):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    accelerator = Accelerator(mixed_precision=("no" if precision == "fp32" else precision))
    device = accelerator.device

    dtype_map = {
        "no": torch.float32,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    train_dtype = dtype_map[precision]

    # ---- experiment dirs & logger ----
    if accelerator.is_main_process:
        os.makedirs(results_dir, exist_ok=True)
        experiment_index = len(glob(f"{results_dir}/*"))
        model_string_name = model.__class__.__name__
        run_name = run_name or f"{model_string_name}-{experiment_index:02d}"
        experiment_dir = f"{results_dir}/{run_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
    else:
        logger = logging.getLogger("train")  # no handlers; quiet worker

    if wandb_logging and accelerator.is_main_process:
        if wandb is None:
            logger.warning("wandb=True but package not available. Skipping W&B init.")
        else:
            run_cfg = {
                "precision": precision,
                "use_diffusion": use_diffusion,
                "epochs": epochs,
                "log_every": log_every,
                "optimizer": "AdamW",
                "lr": 1e-4,
                "weight_decay": 0.0,
                "model": model.__class__.__name__,
                "params": sum(p.numel() for p in model.parameters()),
            }
            if wandb_config:
                run_cfg.update(wandb_config)
            
            wandb.init(project=wandb_project, name=run_name, config=run_cfg, dir=experiment_dir)
            # If you want gradients & params logged (can be heavy):
            # wandb.watch(model, log="all", log_freq=log_every)

    if accelerator.is_main_process:
        logger.info(f"Training with precision={precision} (dtype={train_dtype}) on {device}")

    # ---- models ----
    model = model.to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    diffusion = create_diffusion(timestep_respacing="") if use_diffusion else None

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"WiDiT Parameters: {n_params:,}")
        logger.info(f"Training mode: {'diffusion' if use_diffusion else 'supervised (no diffusion)'}")

    # ---- optimizer ----
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    if accelerator.is_main_process:
        logger.info(f"Training dataloader: {len(training_dataloader):,} batches/epoch")
        if validation_dataloader is not None:
            logger.info(f"Validation dataloader: {len(validation_dataloader):,} batches/epoch")

    # ---- prepare with Accelerate ----
    update_ema(ema, model, decay=0)
    model, ema, opt, training_dataloader = accelerator.prepare(model, ema, opt, training_dataloader)
    if validation_dataloader is not None:
        validation_dataloader = accelerator.prepare(validation_dataloader)

    # ---- training loop ----
    train_steps = 0
    log_steps = 0
    running_loss = 0.0
    start_time = time()
    best_val_loss = float("inf")

    if accelerator.is_main_process:
        logger.info(f"Training for {epochs} epoch(s)...")

    for epoch in range(epochs):
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch}...")

        model.train()

        for batch in track(training_dataloader, total=len(training_dataloader), description="Training:"):
            # Expect (x, target)
            if not isinstance(batch, (list, tuple)) or len(batch) not in {2, 3}:
                raise ValueError("Training dataloader must return (x, target) or (x, target, timestep).")
            x, target = batch[0], batch[1]
            timestep = batch[2] - 1 if len(batch) == 3 else None

            x = x.to(device=accelerator.device, dtype=train_dtype, non_blocking=True)
            target = target.to(device=accelerator.device, dtype=train_dtype, non_blocking=True)

            with accelerator.autocast():
                if use_diffusion:
                    timestep = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=accelerator.device)
                    # Mirror your earlier logic: model(input=target, conditioned=x)
                    loss_dict = diffusion.training_losses(model, target, timestep, dict(conditioned=x))
                    loss = loss_dict["loss"].mean()
                else:
                    y = model(x, timestep=timestep)
                    loss = criterion(y, target)

            opt.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            # ---- logging ----
            if train_steps % log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / max(end_time - start_time, 1e-6)
                avg_loss = running_loss / log_steps
                avg_loss_tensor = torch.tensor(avg_loss, device=accelerator.device)
                avg_loss_tensor = accelerator.reduce(avg_loss_tensor, reduction="mean")
                avg_loss_val = avg_loss_tensor.item()

                if accelerator.is_main_process:
                    logger.info(
                        f"(epoch={epoch:03d}, step={train_steps:07d}) "
                        f"train/loss={avg_loss_val:.4f}  speed={steps_per_sec:.2f} steps/s"
                    )
                    if wandb_logging and wandb is not None:
                        wandb.log(
                            {
                                "train/loss": avg_loss_val,
                                "train/steps_per_sec": steps_per_sec,
                                "epoch": epoch,
                                "global_step": train_steps,
                            },
                            step=train_steps,
                        )

                running_loss = 0.0
                log_steps = 0
                start_time = time()


        # ---- validation (end of epoch) ----
        if validation_dataloader is not None:
            val_loss = _run_validation_loop(
                accelerator=accelerator,
                model_for_eval=ema,               # evaluate EMA
                diffusion=diffusion,
                dataloader=validation_dataloader,
                device=accelerator.device,
                dtype=train_dtype,
                use_diffusion=use_diffusion,
                criterion=criterion,
                timestep_seed=0,  # deterministic validation timesteps
            )
            if accelerator.is_main_process:
                logger.info(f"(epoch={epoch:03d}) val/loss={val_loss:.4f}")
                if wandb_logging and wandb is not None:
                    wandb.log({"val/loss": val_loss, "epoch": epoch, "global_step": train_steps}, step=train_steps)

                # Save best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = f"{checkpoint_dir}/best.pt"
                    ema.save(best_path)
                    logger.info(f"New best checkpoint (val_loss={val_loss:.4f}) saved to {best_path}")

                    if wandb_logging and wandb is not None and wandb_log_artifacts:
                        art = wandb.Artifact(f"{model_string_name}-best", type="model")
                        art.add_file(best_path)
                        wandb.log_artifact(art)

    # ---- final checkpoint ----
    if accelerator.is_main_process:
        if wandb_logging and wandb is not None:
            wandb.summary["best/val_loss"] = best_val_loss
            wandb.finish()
