from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from ssim_torch import ssim


def load_mask(path, device="cpu"):
    content = loadmat(path)
    if "mask_3d_shift" not in content:
        raise ValueError("Mask MAT file must contain 'mask_3d_shift'.")
    array = np.asarray(content["mask_3d_shift"], dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 28 or (not np.isfinite(array).all()):
        raise ValueError("Expected a finite H x (W + 54) x 28 shifted mask.")
    return torch.from_numpy(array.copy()).permute(2, 0, 1)[None].to(device)


def load_truth(path):
    array = np.asarray(loadmat(path)["img"], dtype=np.float32)
    if array.shape != (256, 256, 28):
        raise ValueError(f"{Path(path).name}: expected 'img' with shape (256, 256, 28).")
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError("Simulation truth must contain finite intensities in [0, 1].")
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def gen_meas_torch(data_batch, mask):
    batch, channels, height, width = data_batch.shape
    expected = (1, channels, height, width + 2 * (channels - 1))
    if tuple(mask.shape) != expected:
        raise ValueError(f"Expected mask shape {expected}, got {tuple(mask.shape)}.")
    shifted = data_batch.new_zeros(batch, channels, height, expected[-1])
    shifted[:, :, :, :width] = data_batch
    for channel in range(channels):
        shifted[:, channel] = torch.roll(shifted[:, channel], shifts=2 * channel, dims=2)
    return (shifted * mask).sum(1) / channels * 2


def torch_psnr(image, reference):
    image = (image * 256).round()
    reference = (reference * 256).round()
    values = [
        10 * torch.log10(255 * 255 / torch.mean((a - b) ** 2)) for a, b in zip(image, reference)
    ]
    return torch.stack(values).mean()


def torch_ssim(image, reference):
    return ssim(image[None], reference[None])
