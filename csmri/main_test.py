import argparse
import csv
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from scipy.io import loadmat
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms.functional import center_crop

from moe_duns import load_sampling_matrix
from register_models import dumoe

ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate DUMoE on real-valued, normalized CS-MRI magnitude images."
    )
    parser.add_argument("--model", choices=("dumoe",), default="dumoe")
    parser.add_argument("--ratio", type=int, choices=(5, 10, 20, 30, 40), default=10)
    parser.add_argument("--input_size", type=int, choices=(256,), default=256)
    parser.add_argument("--mask_type", choices=("Radial",), default="Radial")
    parser.add_argument("--dataset", default="Brain_test")
    parser.add_argument("--data_dir", type=Path, help="Override the test image directory")
    parser.add_argument("--mask_path", type=Path, help="Override the mask MAT file")
    parser.add_argument("--checkpoint", type=Path, help="Override the pretrained checkpoint")
    parser.add_argument(
        "--mat_key", default="data", help="Normalized real-valued image array in a MAT file"
    )
    parser.add_argument("--result_dir", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def read_image(path, mat_key, size):
    if path.suffix.lower() == ".mat":
        content = loadmat(path)
        if mat_key not in content:
            raise ValueError(f"{path.name} has no '{mat_key}' array.")
        image = np.asarray(content[mat_key]).squeeze()
        if np.iscomplexobj(image):
            raise ValueError("Use a real-valued magnitude image, normalized to [0, 1].")
    else:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read image: {path.name}")
        image = image.astype(np.float32) / 255.0
    if image.ndim != 2 or min(image.shape) < size:
        raise ValueError(f"{path.name} must be a 2D image of at least {size} x {size}.")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError(f"{path.name} must contain finite intensities in [0, 1].")
    return center_crop(torch.as_tensor(image.astype(np.float32)), [size, size])


def main(args):
    torch.manual_seed(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    data_dir = args.data_dir if args.data_dir else ROOT / "data" / "test" / args.dataset
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing test directory: {data_dir}")
    paths = sorted((p for p in data_dir.iterdir() if p.suffix.lower() in {".png", ".mat"}))
    if not paths:
        raise ValueError("No PNG or MAT test images found.")
    mask = load_sampling_matrix(args.ratio, args.input_size, args.mask_type, args.mask_path).to(
        device
    )
    model = dumoe(args.ratio, pretrained=True, checkpoint_path=args.checkpoint).to(device).eval()
    result_dir = args.result_dir / args.model / data_dir.name / str(args.ratio)
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for index, path in enumerate(paths, 1):
            original = read_image(path, args.mat_key, args.input_size)
            inputs = original[None, None].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter()
            output = model(inputs, mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = perf_counter() - start
            reconstruction = output[0, 0].cpu().numpy().clip(0, 1)
            truth = original.numpy()
            psnr = peak_signal_noise_ratio(truth, reconstruction, data_range=1.0)
            ssim = structural_similarity(truth, reconstruction, data_range=1.0)
            rows.append([path.name, float(psnr), float(ssim), elapsed])
            image = (reconstruction * 255).astype(np.uint8)
            if not cv2.imwrite(str(result_dir / f"{path.name}.reconstructed.png"), image):
                raise OSError("Failed to write reconstructed image.")
            print(
                f"[{index}/{len(paths)}] {path.name}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, time={elapsed:.4f}s"
            )
    means = np.mean([row[1:] for row in rows], axis=0)
    with (result_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "PSNR", "SSIM", "time_seconds"])
        writer.writerows(rows)
        writer.writerow(["average", *means])
    print(f"Average: PSNR={means[0]:.2f}, SSIM={means[1]:.4f}, time={means[2]:.4f}s")
    print(f"Results saved to {result_dir}")


if __name__ == "__main__":
    main(build_parser().parse_args())
