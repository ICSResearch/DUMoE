from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


class CustomDataset(Dataset):
    def __init__(self, file_path, transform):
        folder = Path(file_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing image directory: {folder}")
        self.images = sorted(
            (
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
            )
        )
        if not self.images:
            raise ValueError("No supported training or validation images found.")
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with Image.open(self.images[index]) as source:
            image = self.transform(source.convert("RGB"))
        return (image, image)


def build_dataset(is_train, args):
    if is_train:
        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.RandomResizedCrop((args.input_size, args.input_size)),
                transforms.RandomRotation(degrees=45),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((args.input_size, args.input_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )
    return CustomDataset(args.data_path if is_train else args.eval_data_path, transform)
