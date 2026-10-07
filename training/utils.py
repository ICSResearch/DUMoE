# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: MIT
# License: licenses/ConvNeXt-V2-MIT.txt

import argparse
import builtins
import math
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"yes", "true", "t", "y", "1"}:
        return True
    if value.lower() in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ.get("SLURM_NTASKS", args.world_size))
        args.gpu = args.rank % max(1, torch.cuda.device_count())
    else:
        args.distributed = False
        return
    args.distributed = True
    os.environ.update(
        RANK=str(args.rank), WORLD_SIZE=str(args.world_size), LOCAL_RANK=str(args.gpu)
    )
    backend = "gloo"
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_device(args.gpu)
        args.device = f"cuda:{args.gpu}"
        backend = "nccl"
    dist.init_process_group(
        backend=backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    if args.rank != 0:
        builtins.print = lambda *args, **kwargs: None


class SmoothedValue:
    def __init__(self, fmt="{global_avg:.4f}"):
        self.total, self.count, self.value = 0.0, 0, 0.0
        self.fmt = fmt

    def update(self, value, n=1):
        self.value = float(value)
        self.total += self.value * n
        self.count += n

    @property
    def global_avg(self):
        return self.total / self.count if self.count else 0.0

    def synchronize(self):
        if is_distributed():
            device = "cuda" if dist.get_backend() == "nccl" else "cpu"
            values = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
            dist.all_reduce(values)
            self.count, self.total = values.tolist()

    def __str__(self):
        return self.fmt.format(value=self.value, global_avg=self.global_avg)


class MetricLogger:
    def __init__(self, delimiter="  "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **values):
        for name, value in values.items():
            if value is not None:
                self.meters[name].update(float(value))

    def __getattr__(self, name):
        if name in self.meters:
            return self.meters[name]
        raise AttributeError(name)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize()

    def log_every(self, iterable, frequency, header):
        for index, item in enumerate(iterable):
            yield item
            if index % frequency == 0 or index + 1 == len(iterable):
                values = self.delimiter.join(
                    f"{name}: {meter}" for name, meter in self.meters.items()
                )
                print(f"{header} [{index + 1}/{len(iterable)}] {values}")


class TensorboardLogger:
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0

    def update(self, head="scalar", step=None, **values):
        for name, value in values.items():
            self.writer.add_scalar(
                f"{head}/{name}", float(value), self.step if step is None else step
            )

    def set_step(self):
        self.step += 1

    def flush(self):
        self.writer.flush()


class NativeScalerWithGradNormCount:
    def __init__(self, enabled=False):
        self.scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, update_grad=True):
        self.scaler.scale(loss).backward()
        if not update_grad:
            return None
        self.scaler.unscale_(optimizer)
        parameters = [parameter for parameter in parameters if parameter.grad is not None]
        norm = (
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            if clip_grad is not None
            else torch.norm(
                torch.stack([parameter.grad.detach().norm(2) for parameter in parameters]), 2
            )
        )
        self.scaler.step(optimizer)
        self.scaler.update()
        return norm

    def state_dict(self):
        return self.scaler.state_dict()

    def load_state_dict(self, state):
        self.scaler.load_state_dict(state)


def save_model(args, epoch, model, optimizer, scaler, ema=None):
    if get_rank() != 0:
        return
    folder = Path(args.output_dir)
    state = ema.ema.state_dict() if epoch == "best-ema" and ema is not None else model.state_dict()
    checkpoint = {
        "model": state,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
    }
    if ema is not None:
        checkpoint["model_ema"] = ema.ema.state_dict()
    torch.save(checkpoint, folder / f"checkpoint-{args.model}-{args.cs_ratio}-{epoch}.pth")
    if isinstance(epoch, int):
        old_epoch = epoch - args.save_ckpt_num * args.save_ckpt_freq
        old = folder / f"checkpoint-{args.model}-{args.cs_ratio}-{old_epoch}.pth"
        old.unlink(missing_ok=True)


def auto_load_model(args, model, optimizer, scaler, ema=None):
    if args.auto_resume and not args.resume:
        paths = Path(args.output_dir).glob(f"checkpoint-{args.model}-{args.cs_ratio}-*.pth")
        epochs = [
            int(path.stem.rsplit("-", 1)[-1])
            for path in paths
            if path.stem.rsplit("-", 1)[-1].isdigit()
        ]
        if epochs:
            args.resume = str(
                Path(args.output_dir) / f"checkpoint-{args.model}-{args.cs_ratio}-{max(epochs)}.pth"
            )
    if not args.resume:
        return False
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in checkpoint["model"].items()}
    model.load_state_dict(state, strict=True)
    if ema is not None:
        ema.ema.load_state_dict(checkpoint.get("model_ema", state), strict=True)
    if not args.eval and "optimizer" in checkpoint and "epoch" in checkpoint:
        if not isinstance(checkpoint["epoch"], int):
            raise ValueError("Use --init_checkpoint for best checkpoints.")
        optimizer.load_state_dict(checkpoint["optimizer"])
        args.start_epoch = checkpoint["epoch"] + 1
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
    return True


def adjust_learning_rate(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        rate = args.lr * epoch / args.warmup_epochs
    else:
        rate = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1
            + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))
        )
    for group in optimizer.param_groups:
        group["lr"] = rate * group.get("lr_scale", 1.0)
