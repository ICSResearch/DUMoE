import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import trange
from training_memory import enable_gradient_checkpointing

from architecture import model_generator
from losses import MixedLoss
from training_data import load_training, shuffle_crop
from utils import gen_meas_torch, load_mask, load_truth, torch_psnr

ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train DUMoE on simulated SCI using CAVE crops and KAIST validation."
    )
    parser.add_argument("--data_root", type=Path, default=ROOT / "data" / "SCI")
    parser.add_argument("--outf", type=Path, default=ROOT / "runs" / "dumoe")
    parser.add_argument(
        "--pretrained_model_path", type=Path, help="Optional weights for initialization"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Effective batch size; samples run sequentially"
    )
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--epoch_sam_num", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=0.00016)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint_gradients",
        action="store_true",
        help="Recompute stage activations to reduce training memory",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(args):
    if (
        args.batch_size < 1
        or args.epoch_sam_num < args.batch_size
        or args.max_epoch <= 5
        or (args.learning_rate <= 0)
    ):
        raise ValueError(
            "Use positive batch size and learning rate, at least one batch per epoch, and more than five epochs."
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    images = load_training(args.data_root / "cave_1024_28")
    data_dir = args.data_root / "TSA_simu_data"
    mask = load_mask(data_dir / "mask_3d_shift.mat", device)
    truth_paths = sorted((data_dir / "Truth").glob("*.mat"))
    if not truth_paths:
        raise FileNotFoundError("No validation truth MAT files found.")
    truths = [load_truth(path) for path in truth_paths]
    model = model_generator(pretrained_model_path=args.pretrained_model_path, device=device)
    if args.checkpoint_gradients:
        enable_gradient_checkpointing(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    criterion = MixedLoss().to(device)
    args.outf.mkdir(parents=True, exist_ok=True)
    if (args.outf / "latest.pth").exists() or (args.outf / "best.pth").exists():
        raise FileExistsError("Output directory already contains checkpoints; choose a new --outf.")
    best = -float("inf")
    batches = args.epoch_sam_num // args.batch_size
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        total = 0.0
        for step in trange(batches, desc=f"Epoch {epoch}"):
            progress = epoch + step / batches
            lr = (
                args.learning_rate * progress / 5
                if progress < 5
                else 1e-06
                + (args.learning_rate - 1e-06)
                * 0.5
                * (1 + math.cos(math.pi * (progress - 5) / (args.max_epoch - 5)))
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            ground_truth = shuffle_crop(images, args.batch_size).to(device)
            optimizer.zero_grad(set_to_none=True)
            for truth in ground_truth.split(1):
                measurement = gen_meas_torch(truth, mask)
                prediction, auxiliary = model(measurement, mask)
                loss = criterion(prediction, truth, auxiliary) / args.batch_size
                loss.backward()
                total += loss.item()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            scores = []
            for truth in truths:
                truth = truth.to(device)[None]
                prediction = model(gen_meas_torch(truth, mask), mask)
                scores.append(torch_psnr(prediction[0], truth[0]).item())
        score = float(np.mean(scores))
        state = {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
        torch.save(state, args.outf / "latest.pth")
        if score > best:
            best = score
            torch.save(state, args.outf / "best.pth")
        print(f"Epoch {epoch}: loss={total / batches:.6f}, validation PSNR={score:.2f}")


if __name__ == "__main__":
    main(build_parser().parse_args())
