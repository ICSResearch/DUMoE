import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat

from architecture import model_generator

ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(
        description="Reconstruct real hyperspectral snapshot measurements with DUMoE."
    )
    parser.add_argument("--method", choices=("dumoe",), default="dumoe")
    parser.add_argument("--data_root", type=Path, default=ROOT / "data" / "SCI")
    parser.add_argument(
        "--pretrained_model_path", type=Path, default=ROOT / "model" / "dumoe_sci_real.pth"
    )
    parser.add_argument("--outf", type=Path, default=ROOT / "results" / "dumoe")
    parser.add_argument("--input_setting", choices=("Y",), default="Y")
    parser.add_argument("--input_mask", choices=("Phi",), default="Phi")
    parser.add_argument("--height", type=int, default=660)
    parser.add_argument("--width", type=int, default=660)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--gpu_id", help="Optional CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def load_mask(path, height, width, device):
    array = np.asarray(loadmat(path)["mask_3d_shift"], dtype=np.float32)
    if (
        array.ndim != 3
        or array.shape[2] != 28
        or array.shape[0] < height
        or (array.shape[1] < width + 54)
    ):
        raise ValueError("Mask must cover H x (W + 54) x 28.")
    array = array[:height, : width + 54]
    if not np.isfinite(array).all():
        raise ValueError("Mask contains non-finite values.")
    return torch.from_numpy(array.copy()).permute(2, 0, 1)[None].to(device)


def load_measurement(path, height, width, device):
    array = np.asarray(loadmat(path)["meas_real"], dtype=np.float32)
    if (
        array.ndim != 2
        or array.shape[0] < height
        or array.shape[1] < width + 54
        or (not np.isfinite(array).all())
    ):
        raise ValueError("Expected finite 'meas_real' covering H x (W + 54).")
    array = array[:height, : width + 54].clip(0, 1)
    array = array / (array.max() + 1e-07) * 0.9
    return torch.from_numpy(array.copy())[None].to(device)


def main(args):
    if args.height != args.width or args.height < 8 or args.height % 4:
        raise ValueError("The supplied model requires square spatial crops divisible by four.")
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    torch.manual_seed(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    data_dir = args.data_root / "TSA_real_data"
    paths = sorted(
        (data_dir / "Measurements").glob("scene*.mat"),
        key=lambda p: int(re.search("\\d+$", p.stem).group()) if re.search("\\d+$", p.stem) else 0,
    )
    if not paths:
        raise FileNotFoundError("No scene MAT files found under TSA_real_data/Measurements.")
    mask = load_mask(data_dir / "mask_3d_shift.mat", args.height, args.width, device)
    model = model_generator(args.method, args.pretrained_model_path, device).eval()
    args.outf.mkdir(parents=True, exist_ok=True)
    predictions = []
    with torch.inference_mode():
        for path in paths:
            measurement = load_measurement(path, args.height, args.width, device)
            prediction = model(measurement, mask).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            predictions.append(prediction)
            savemat(args.outf / f"{path.stem}.mat", {"res": prediction})
            print(f"Reconstructed {path.name}")
    savemat(args.outf / "Real_result.mat", {"pred": np.stack(predictions)})
    print(f"Results saved to {args.outf}")


if __name__ == "__main__":
    main(build_parser().parse_args())
