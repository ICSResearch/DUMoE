import random
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


def load_training(path, cave=False):
    paths = sorted(Path(path).glob("*.mat"))
    if not paths:
        raise FileNotFoundError("No training MAT files found.")
    images = []
    for path in paths:
        content = loadmat(path)
        if cave:
            key = "img_expand" if "img_expand" in content else "img"
            scale = 65535.0 if key == "img_expand" else 65536.0
        else:
            key, scale = ("HSI", 1.0)
        image = np.asarray(content[key], dtype=np.float32) / scale
        if image.ndim != 3 or image.shape[2] != 28 or (not np.isfinite(image).all()):
            raise ValueError("Training cubes must be finite H x W x 28 arrays.")
        images.append(image.clip(0, 1))
    return images


class TrainingDataset(Dataset):
    def __init__(self, cave, kaist, mask_path, size=320, samples=2222):
        self.cave, self.kaist = (cave, kaist)
        self.size, self.samples = (size, samples)
        self.mask = np.asarray(loadmat(mask_path)["mask_3d_shift"], dtype=np.float32)
        if size < 8 or size % 4 or samples < 1:
            raise ValueError("Use positive samples and a crop size divisible by four.")
        if self.mask.ndim != 3 or self.mask.shape[2] != 28 or (not np.isfinite(self.mask).all()):
            raise ValueError("Mask must be a finite H x (W + 54) x 28 array.")
        if self.mask.shape[0] < size or self.mask.shape[1] < size + 54:
            raise ValueError("Mask is smaller than the requested crop.")
        if not cave or not kaist or any((min(image.shape[:2]) < size for image in cave + kaist)):
            raise ValueError("Both training sets must contain cubes covering the requested crop.")

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        pool = self.cave if random.randint(0, 1) == 0 else self.kaist
        image = random.choice(pool)
        height, width = image.shape[:2]
        x = random.randint(0, height - self.size)
        y = random.randint(0, width - self.size)
        label = image[x : x + self.size, y : y + self.size]
        x = random.randint(0, self.mask.shape[0] - self.size)
        crop_width = min(self.mask.shape[1] - 54, self.mask.shape[0])
        if crop_width < self.size + 54:
            crop_width = self.mask.shape[1]
        y = random.randint(0, crop_width - self.size - 54)
        mask = self.mask[x : x + self.size, y : y + self.size + 54]
        label = np.rot90(label, random.randint(0, 3))
        if random.randint(0, 1):
            label = label[:, ::-1]
        if random.randint(0, 1):
            label = label[::-1]
        shifted = np.zeros((self.size, self.size + 54, 28), dtype=np.float32)
        shifted[:, : self.size] = label
        for channel in range(28):
            shifted[:, :, channel] = np.roll(shifted[:, :, channel], 2 * channel, axis=1)
        measurement = (shifted * mask).sum(2)
        measurement = measurement / (measurement.max() + 1e-07) * 0.9
        quantum_efficiency, levels = (0.4, 2048)
        measurement = np.random.binomial(
            (measurement * levels / quantum_efficiency).astype(int), quantum_efficiency
        )
        measurement = measurement.astype(np.float32) / levels
        return (
            torch.from_numpy(measurement.copy()),
            torch.from_numpy(label.copy()).permute(2, 0, 1),
            torch.from_numpy(mask.copy()).permute(2, 0, 1),
        )
