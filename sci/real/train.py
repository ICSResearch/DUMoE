import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from training_memory import enable_gradient_checkpointing

from architecture import model_generator
from losses import MixedLoss
from training_data import TrainingDataset, load_training

ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune real-SCI DUMoE on CAVE/KAIST with the sensor noise model."
    )
    parser.add_argument("--data_root", type=Path, default=ROOT / "data" / "SCI")
    parser.add_argument("--outf", type=Path, default=ROOT / "runs" / "dumoe")
    parser.add_argument(
        "--pretrained_model_path",
        type=Path,
        required=True,
        help="Simulation checkpoint for initialization",
    )
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Effective batch size; samples run sequentially"
    )
    parser.add_argument("--max_epoch", type=int, default=400)
    parser.add_argument(
        "--epoch_sam_num",
        type=int,
        help="Samples per epoch; default follows the original crop-size rule",
    )
    parser.add_argument("--learning_rate", type=float, default=4e-05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint_gradients",
        action="store_true",
        help="Recompute stage activations to reduce training memory",
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main(args):
    if args.batch_size < 1 or args.max_epoch <= 5 or args.size < 96 or (args.learning_rate <= 0):
        raise ValueError(
            "Use a positive batch size and learning rate, crop size >= 96, and more than five epochs."
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cave = load_training(args.data_root / "cave_1024_28", cave=True)
    kaist = load_training(args.data_root / "KAIST_CVPR2021")
    samples = (
        args.epoch_sam_num if args.epoch_sam_num is not None else 20000 // (args.size // 96) ** 2
    )
    dataset = TrainingDataset(
        cave, kaist, args.data_root / "TSA_real_data" / "mask_3d_shift.mat", args.size, samples
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = model_generator(pretrained_model_path=args.pretrained_model_path, device=device)
    if args.checkpoint_gradients:
        enable_gradient_checkpointing(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    criterion = MixedLoss().to(device)
    args.outf.mkdir(parents=True, exist_ok=True)
    if (args.outf / "latest.pth").exists():
        raise FileExistsError(
            "Output directory already contains a checkpoint; choose a new --outf."
        )
    for epoch in range(args.max_epoch):
        model.train()
        total = 0.0
        for step, (measurement, label, mask) in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}")):
            progress = epoch + step / len(loader)
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
            optimizer.zero_grad(set_to_none=True)
            for meas, truth, phi in zip(measurement.split(1), label.split(1), mask.split(1)):
                prediction, auxiliary = model(meas.to(device), phi.to(device))
                loss = criterion(prediction, truth.to(device), auxiliary) / len(measurement)
                loss.backward()
                total += loss.item()
            optimizer.step()
        state = {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
        torch.save(state, args.outf / "latest.pth")
        print(f"Epoch {epoch + 1}: loss={total / len(loader):.6f}")


if __name__ == "__main__":
    main(build_parser().parse_args())
