# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: MIT
# License: licenses/ConvNeXt-V2-MIT.txt

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

from . import utils
from .engine import evaluate, train_one_epoch
from .optim import create_optimizer


def get_args_parser(task, root):
    parser = argparse.ArgumentParser(description=f"DUMoE {task.upper()} training")
    parser.add_argument("--model", default="dumoe", choices=("dumoe",))
    parser.add_argument("--batch_size", default=32, type=int, help="Effective batch per process")
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--update_freq", default=1, type=int, help="Gradient accumulation steps")
    parser.add_argument("--input_size", default=96 if task == "ics" else 320, type=int)
    parser.add_argument(
        "--cs_ratio",
        default=10,
        type=int,
        choices=(1, 4, 10, 25, 30, 40, 50) if task == "ics" else (5, 10, 20, 30, 40),
    )
    parser.add_argument("--opt", default="adamw", choices=("adamw", "adam", "sgd"))
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--blr", default=2e-4, type=float, help="lr = blr * update_freq * world_size"
    )
    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--opt_eps", default=1e-8, type=float)
    parser.add_argument("--opt_betas", type=float, nargs=2)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--clip_grad", type=float)
    parser.add_argument("--use_amp", type=utils.str2bool, default=False)
    parser.add_argument("--model_ema", type=utils.str2bool, default=False)
    parser.add_argument("--model_ema_decay", type=float, default=0.9999)
    parser.add_argument("--model_ema_force_cpu", type=utils.str2bool, default=False)
    parser.add_argument("--model_ema_eval", type=utils.str2bool, default=False)
    parser.add_argument("--data_path", type=Path, default=root / "data" / "train")
    parser.add_argument("--eval_data_path", type=Path, default=root / "data" / "val")
    parser.add_argument("--output_dir", default=root / "runs" / "dumoe")
    parser.add_argument("--log_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--init_checkpoint", type=Path)
    parser.add_argument("--resume", default="")
    parser.add_argument("--auto_resume", type=utils.str2bool, default=False)
    parser.add_argument("--save_ckpt", type=utils.str2bool, default=True)
    parser.add_argument("--save_ckpt_freq", default=5, type=int)
    parser.add_argument("--save_ckpt_num", default=10, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", type=utils.str2bool, default=False)
    parser.add_argument("--disable_eval", type=utils.str2bool, default=False)
    parser.add_argument("--dist_eval", type=utils.str2bool, default=False)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--pin_mem", type=utils.str2bool, default=True)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_on_itp", type=utils.str2bool, default=False)
    parser.add_argument("--dist_url", default="env://")
    if task == "csmri":
        parser.add_argument("--mask_type", default="Radial", choices=("Radial", "Cartesian"))
        parser.add_argument("--mask_path", type=Path)
    return parser


def validate_args(args, task):
    if min(args.batch_size, args.update_freq, args.save_ckpt_freq, args.save_ckpt_num) < 1:
        raise ValueError("Batch, accumulation, and checkpoint counts must be positive.")
    if min(args.warmup_epochs, args.num_workers, args.start_epoch) < 0:
        raise ValueError("Warm-up, worker, and starting-epoch counts must be nonnegative.")
    if not args.eval and args.epochs <= args.warmup_epochs:
        raise ValueError("Epochs must exceed warm-up epochs.")
    block = 32 if task == "ics" else 4
    if args.input_size < (32 if task == "ics" else 8) or args.input_size % block:
        raise ValueError(f"Crop size must be a positive multiple of {block}.")
    if args.eval and args.disable_eval:
        raise ValueError("Evaluation requires validation data.")
    if args.model_ema_eval and not args.model_ema:
        raise ValueError("EMA evaluation requires --model_ema true.")
    if args.use_amp and torch.device(args.device).type != "cuda":
        raise ValueError("AMP requires CUDA.")
    if args.init_checkpoint and (args.resume or args.auto_resume):
        raise ValueError("Initialization and resume are mutually exclusive.")
    if args.auto_resume and not args.output_dir:
        raise ValueError("Automatic resume requires an output directory.")
    if args.lr is not None and args.lr <= 0 or args.blr <= 0 or args.opt_eps <= 0:
        raise ValueError("Learning rates and optimizer epsilon must be positive.")
    if args.min_lr < 0 or args.weight_decay < 0:
        raise ValueError("Minimum learning rate and weight decay must be nonnegative.")
    if args.clip_grad is not None and args.clip_grad <= 0:
        raise ValueError("Gradient clipping norm must be positive.")
    if not 0 <= args.model_ema_decay < 1:
        raise ValueError("EMA decay must be in [0, 1).")
    if args.opt_betas and any(not 0 <= beta < 1 for beta in args.opt_betas):
        raise ValueError("Optimizer betas must be in [0, 1).")
    if args.opt == "sgd" and args.momentum <= 0:
        raise ValueError("Nesterov SGD requires positive momentum.")


def run(args, task, dataset_factory, model_factory, mask_loader=None):
    validate_args(args, task)
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    rank, world_size = utils.get_rank(), utils.get_world_size()
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = device.type == "cuda"
    train_data = dataset_factory(True, args) if not args.eval else None
    val_data = dataset_factory(False, args) if not args.disable_eval else None
    loader_args = dict(
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem
    )
    train_loader = None
    if train_data is not None:
        sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, seed=args.seed)
        train_loader = DataLoader(train_data, sampler=sampler, drop_last=True, **loader_args)
        if not len(train_loader):
            raise ValueError("Training data must contain at least one complete batch.")
    val_loader = None
    if val_data is not None:
        sampler = (
            DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False)
            if args.dist_eval
            else SequentialSampler(val_data)
        )
        if args.dist_eval and len(val_data) % world_size:
            print("Distributed validation pads the dataset with repeated samples.")
        val_loader = DataLoader(val_data, sampler=sampler, **loader_args)
    mask = (
        mask_loader(args.cs_ratio, args.input_size, args.mask_type, args.mask_path).to(device)
        if mask_loader
        else None
    )
    model = model_factory(args.cs_ratio).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint)
        model.load_state_dict(
            {key.removeprefix("module."): value for key, value in state.items()}, strict=True
        )
    ema = None
    if args.model_ema:
        from timm.utils import ModelEma

        ema = ModelEma(
            model, decay=args.model_ema_decay, device="cpu" if args.model_ema_force_cpu else ""
        )
    args.lr = args.lr if args.lr is not None else args.blr * args.update_freq * world_size
    optimizer = create_optimizer(args, model)
    scaler = utils.NativeScalerWithGradNormCount(enabled=args.use_amp)
    resumed = utils.auto_load_model(args, model, optimizer, scaler, ema)
    if args.eval and not (args.init_checkpoint or resumed):
        raise ValueError("Evaluation requires initialization or resume weights.")
    if not args.eval and not resumed and args.output_dir:
        if list(Path(args.output_dir).glob(f"checkpoint-{args.model}-{args.cs_ratio}-*.pth")):
            raise FileExistsError("Choose a new output directory or resume an existing checkpoint.")
    if args.eval:
        evaluate(val_loader, model, device, args.use_amp, mask)
        return
    if args.start_epoch >= args.epochs:
        raise ValueError("Starting epoch must be below the requested total epochs.")
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    writer = utils.TensorboardLogger(args.log_dir) if rank == 0 and args.log_dir else None
    bare_model = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    criterion = torch.nn.MSELoss()
    best, best_ema = -float("inf"), -float("inf")
    try:
        for epoch in range(args.start_epoch, args.epochs):
            train_loader.sampler.set_epoch(epoch)
            stats = train_one_epoch(
                model,
                criterion,
                train_loader,
                optimizer,
                device,
                epoch,
                scaler,
                args.clip_grad,
                ema,
                writer,
                args,
                mask,
            )
            log = {"epoch": epoch, **{f"train_{key}": value for key, value in stats.items()}}
            if (
                args.output_dir
                and args.save_ckpt
                and ((epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs)
            ):
                utils.save_model(args, epoch, bare_model, optimizer, scaler, ema)
            if val_loader is not None:
                stats = evaluate(val_loader, model, device, args.use_amp, mask)
                log.update({f"test_{key}": value for key, value in stats.items()})
                if stats["psnr"] > best:
                    best = stats["psnr"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(args, "best", bare_model, optimizer, scaler, ema)
                if writer:
                    writer.update(head="validation", step=epoch, **stats)
                if args.model_ema_eval:
                    stats = evaluate(
                        val_loader, ema.ema, next(ema.ema.parameters()).device, args.use_amp, mask
                    )
                    log.update({f"test_{key}_ema": value for key, value in stats.items()})
                    if stats["psnr"] > best_ema:
                        best_ema = stats["psnr"]
                        if args.output_dir and args.save_ckpt:
                            utils.save_model(args, "best-ema", bare_model, optimizer, scaler, ema)
            if args.output_dir and rank == 0:
                with (Path(args.output_dir) / f"{args.model}_{args.cs_ratio}_log.txt").open(
                    "a", encoding="utf-8"
                ) as stream:
                    values = {
                        key: value if np.isfinite(value) else str(value)
                        for key, value in log.items()
                    }
                    stream.write(json.dumps(values, allow_nan=False) + "\n")
            if writer:
                writer.flush()
    finally:
        if writer:
            writer.writer.close()
