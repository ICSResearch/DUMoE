# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: MIT
# License: licenses/ConvNeXt-V2-MIT.txt

import math
from contextlib import nullcontext

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio

from . import utils
from .utils import adjust_learning_rate


def amp_context(device, enabled):
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if enabled and device.type == "cuda"
        else nullcontext()
    )


def train_one_epoch(
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler,
    max_norm=None,
    model_ema=None,
    log_writer=None,
    args=None,
    mask=None,
):
    if len(data_loader) == 0:
        raise ValueError("No complete training batches were loaded.")
    model.train()
    logger = utils.MetricLogger(delimiter="  ")
    logger.add_meter("lr", utils.SmoothedValue(fmt="{value:.6f}"))
    optimizer.zero_grad(set_to_none=True)
    mask = mask.to(device) if mask is not None else None
    auxiliary_weight = 1.0 if mask is not None else 0.001
    batches = len(data_loader)
    for step, (samples, targets) in enumerate(logger.log_every(data_loader, 50, f"Epoch {epoch}")):
        window_start = step // args.update_freq * args.update_freq
        window_size = min(args.update_freq, batches - window_start)
        if step % args.update_freq == 0:
            adjust_learning_rate(optimizer, epoch + step / batches, args)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        update = (step + 1) % args.update_freq == 0 or step + 1 == batches
        predictions, losses = ([], [])
        grad_norm = None
        for index, (sample, target) in enumerate(zip(samples.split(1), targets.split(1))):
            with amp_context(device, args.use_amp):
                prediction, auxiliary = model(sample) if mask is None else model(sample, mask)
                loss = criterion(prediction, target) + auxiliary_weight * auxiliary
            if not math.isfinite(loss.item()):
                raise FloatingPointError("Training loss is not finite.")
            predictions.append(prediction.detach())
            losses.append(loss.item())
            scaled_loss = loss / (len(samples) * window_size)
            final_sample = index + 1 == len(samples)
            if args.use_amp:
                grad_norm = loss_scaler(
                    scaled_loss,
                    optimizer,
                    clip_grad=max_norm,
                    parameters=model.parameters(),
                    update_grad=update and final_sample,
                )
            else:
                scaled_loss.backward()
                if update and final_sample:
                    if max_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    optimizer.step()
            if update and final_sample:
                optimizer.zero_grad(set_to_none=True)
                if model_ema is not None:
                    model_ema.update(model)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        output = torch.cat(predictions).float().cpu().numpy()
        truth = targets.float().cpu().numpy()
        psnr = peak_signal_noise_ratio(truth, output, data_range=1.0)
        lr = max((group["lr"] for group in optimizer.param_groups))
        loss_value = float(np.mean(losses))
        logger.update(loss=loss_value, psnr=float(psnr), lr=lr)
        if grad_norm is not None:
            logger.update(grad_norm=float(grad_norm))
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(psnr=float(psnr), head="psnr")
            log_writer.update(lr=lr, head="opt")
            log_writer.set_step()
    logger.synchronize_between_processes()
    return {key: meter.global_avg for key, meter in logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, mask=None):
    if data_loader is None or len(data_loader) == 0:
        raise ValueError("No validation images were loaded.")
    model.eval()
    criterion = torch.nn.MSELoss()
    mask = mask.to(device) if mask is not None else None
    logger = utils.MetricLogger(delimiter="  ")
    for images, targets in logger.log_every(data_loader, 10, "Validation"):
        for image, target in zip(images.split(1), targets.split(1)):
            image, target = (image.to(device), target.to(device))
            with amp_context(device, use_amp):
                prediction = model(image) if mask is None else model(image, mask)
                loss = criterion(prediction, target)
            psnr = peak_signal_noise_ratio(
                target.float().cpu().numpy(), prediction.float().cpu().numpy(), data_range=1.0
            )
            logger.update(loss=float(loss), psnr=float(psnr))
    logger.synchronize_between_processes()
    print(f"Validation PSNR={logger.psnr.global_avg:.4f}, MSE={logger.loss.global_avg:.6f}")
    return {key: meter.global_avg for key, meter in logger.meters.items()}
