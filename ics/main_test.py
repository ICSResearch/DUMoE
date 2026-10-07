import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from register_models import dumoe

ROOT = Path(__file__).resolve().parent
RATIOS = (1, 4, 10, 25, 30, 40, 50)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate DUMoE on luminance image compressive sensing."
    )
    parser.add_argument("--model", choices=("dumoe",), default="dumoe")
    parser.add_argument(
        "--ratio", type=int, choices=RATIOS, default=25, help="Sampling ratio in percent"
    )
    parser.add_argument("--block_size", type=int, choices=(32,), default=32)
    parser.add_argument("--dataset", default="Set14", help="Dataset name under data/test")
    parser.add_argument("--data_dir", type=Path, help="Override the test image directory")
    parser.add_argument("--checkpoint", type=Path, help="Override the pretrained checkpoint")
    parser.add_argument("--result_dir", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lpips", action="store_true", help="Compute standard VGG LPIPS; may download VGG weights"
    )
    return parser


def imread_CS_py(image, args):
    row, col = image.shape
    padded = np.pad(image, ((0, -row % args.block_size), (0, -col % args.block_size)))
    return (image, row, col, padded, *padded.shape)


def perceptual_input(image, device):
    tensor = torch.as_tensor(image, dtype=torch.float32, device=device)
    return tensor[None, None].repeat(1, 3, 1, 1) * 2 - 1


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    data_dir = args.data_dir if args.data_dir else ROOT / "data" / "test" / args.dataset
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing test directory: {data_dir}")
    paths = sorted(
        (
            p
            for p in data_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        )
    )
    if not paths:
        raise ValueError("No supported test images found.")
    model = dumoe(args.ratio, pretrained=True, checkpoint_path=args.checkpoint).to(device).eval()
    metric = None
    if args.lpips:
        import lpips

        metric = lpips.LPIPS(net="vgg").to(device).eval()
    result_dir = args.result_dir / args.model / data_dir.name / str(args.ratio)
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for index, path in enumerate(paths, 1):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot read image: {path.name}")
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            original, height, width, padded, _, _ = imread_CS_py(yuv[:, :, 0], args)
            if min(height, width) < 7:
                raise ValueError("Images must be at least 7 x 7 pixels for SSIM.")
            inputs = torch.as_tensor(padded / 255.0, dtype=torch.float32, device=device)[None, None]
            reconstruction = model(inputs)[0, 0].cpu().numpy()[:height, :width].clip(0, 1)
            psnr = peak_signal_noise_ratio(
                original.astype(np.float64), reconstruction * 255, data_range=255
            )
            ssim = structural_similarity(
                original.astype(np.float64), reconstruction * 255, data_range=255
            )
            row = [path.name, float(psnr), float(ssim)]
            if metric is not None:
                score = metric(
                    perceptual_input(original / 255.0, device),
                    perceptual_input(reconstruction, device),
                ).item()
                row.append(score)
            rows.append(row)
            yuv[:, :, 0] = reconstruction * 255
            output = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)
            if not cv2.imwrite(str(result_dir / f"{path.name}.reconstructed.png"), output):
                raise OSError("Failed to write reconstructed image.")
            print(f"[{index}/{len(paths)}] {path.name}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")
    means = np.mean([row[1:] for row in rows], axis=0)
    with (result_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "PSNR", "SSIM"] + (["LPIPS"] if metric is not None else []))
        writer.writerows(rows)
        writer.writerow(["average", *means])
    print(f"Average: PSNR={means[0]:.2f}, SSIM={means[1]:.4f}")
    print(f"Results saved to {result_dir}")


if __name__ == "__main__":
    main(build_parser().parse_args())
