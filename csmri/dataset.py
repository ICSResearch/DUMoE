from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms


class FastMRIDataset(Dataset):
    def __init__(self, file_path, transform, input_size):
        folder = Path(file_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing MRI directory: {folder}")
        self.images = sorted(
            (
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in {".mat", ".png"}
            )
        )
        if not self.images:
            raise ValueError("No MAT or PNG magnitude MRI images found.")
        self.transform, self.input_size = (transform, input_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        if path.suffix.lower() == ".mat":
            content = loadmat(path)
            if "data" not in content:
                raise ValueError(f"{path.name} must contain a 'data' array.")
            image = np.asarray(content["data"]).squeeze()
            if np.iscomplexobj(image):
                raise ValueError("MRI training requires real-valued magnitude images.")
            image = image.astype(np.float32)
        else:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.dtype != np.uint8:
                raise ValueError("MRI PNG inputs must be readable 8-bit grayscale images.")
            image = image.astype(np.float32) / 255.0
        if image.ndim != 2 or min(image.shape) < self.input_size:
            raise ValueError("MRI images must be 2D and cover the requested center crop.")
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("MRI intensities must be finite and normalized to [0, 1].")
        image = self.transform(torch.from_numpy(image.copy())[None])
        return (image, image)


def build_dataset(is_train, args):
    steps = [transforms.CenterCrop(args.input_size)]
    if is_train:
        steps.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    return FastMRIDataset(
        args.data_path if is_train else args.eval_data_path,
        transforms.Compose(steps),
        args.input_size,
    )
