import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

from architecture import model_generator
from utils import gen_meas_torch, load_mask, load_truth, torch_psnr, torch_ssim

ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate DUMoE on simulated hyperspectral snapshot measurements."
    )
    parser.add_argument("--method", choices=("dumoe",), default="dumoe")
    parser.add_argument("--data_root", type=Path, default=ROOT / "data" / "SCI")
    parser.add_argument(
        "--pretrained_model_path", type=Path, default=ROOT / "model" / "dumoe_sci_sim.pth"
    )
    parser.add_argument("--outf", type=Path, default=ROOT / "results" / "dumoe")
    parser.add_argument("--input_setting", choices=("Y",), default="Y")
    parser.add_argument("--input_mask", choices=("Phi", "Phi_PhiPhiT"), default="Phi")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--gpu_id", help="Optional CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(args):
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    torch.manual_seed(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    data_dir = args.data_root / "TSA_simu_data"
    paths = sorted((data_dir / "Truth").glob("*.mat"))
    if not paths:
        raise FileNotFoundError("No truth MAT files found under TSA_simu_data/Truth.")
    mask = load_mask(data_dir / "mask_3d_shift.mat", device)
    model = model_generator(args.method, args.pretrained_model_path, device).eval()
    args.outf.mkdir(parents=True, exist_ok=True)
    predictions, references, rows = ([], [], [])
    with torch.inference_mode():
        for path in paths:
            truth = load_truth(path).to(device)
            measurement = gen_meas_torch(truth[None], mask)
            prediction = model(measurement, mask)[0]
            psnr = torch_psnr(prediction, truth).item()
            ssim = torch_ssim(prediction, truth).item()
            rows.append([path.name, psnr, ssim])
            predictions.append(prediction.permute(1, 2, 0).cpu().numpy())
            references.append(truth.permute(1, 2, 0).cpu().numpy())
            print(f"{path.name}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")
    savemat(
        args.outf / f"{args.method}.mat",
        {"truth": np.stack(references), "pred": np.stack(predictions)},
    )
    means = np.mean([row[1:] for row in rows], axis=0)
    with (args.outf / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene", "PSNR", "SSIM"])
        writer.writerows(rows)
        writer.writerow(["average", *means])
    print(f"Average: PSNR={means[0]:.2f}, SSIM={means[1]:.4f}")
    print(f"Results saved to {args.outf}")


if __name__ == "__main__":
    main(build_parser().parse_args())
