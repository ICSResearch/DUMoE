import random
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat


def load_training(path):
    paths = sorted(Path(path).glob("*.mat"))
    if not paths:
        raise FileNotFoundError("No CAVE training MAT files found.")
    images = []
    for path in paths:
        content = loadmat(path)
        key = "img_expand" if "img_expand" in content else "img"
        image = np.asarray(content[key], dtype=np.float32) / 65536.0
        if image.ndim != 3 or image.shape[2] != 28 or min(image.shape[:2]) < 256:
            raise ValueError("CAVE training images must be H x W x 28, at least 256 x 256.")
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Expected CAVE intensities in [0, 65536] before normalization.")
        images.append(image)
    return images


def shuffle_crop(train_data, batch_size, crop_size=256, augment=True):
    if augment:
        flag = random.randint(0, 1)
        if flag:
            index = np.random.choice(range(len(train_data)), batch_size)
            processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
            for i in range(batch_size):
                h, w, _ = train_data[index[i]].shape
                x_index = np.random.randint(0, h - crop_size + 1)
                y_index = np.random.randint(0, w - crop_size + 1)
                processed_data[i, :, :, :] = train_data[index[i]][
                    x_index : x_index + crop_size, y_index : y_index + crop_size, :
                ]
            gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
            for i in range(gt_batch.shape[0]):
                gt_batch[i] = augment_1(gt_batch[i])
        else:
            gt_batch = []
            processed_data = torch.zeros((4, 28, 128, 128)).float()
            for i in range(batch_size):
                sample_list = np.random.randint(0, len(train_data), 4)
                for j in range(4):
                    h, w, _ = train_data[sample_list[j]].shape
                    x_index = np.random.randint(0, h - crop_size // 2 + 1)
                    y_index = np.random.randint(0, w - crop_size // 2 + 1)
                    processed_data[j] = augment_1(
                        torch.from_numpy(
                            np.transpose(
                                train_data[sample_list[j]][
                                    x_index : x_index + crop_size // 2,
                                    y_index : y_index + crop_size // 2,
                                    :,
                                ],
                                (2, 0, 1),
                            )
                        )
                    )
                generated_sample = processed_data
                gt_batch.append(augment_2(generated_sample))
            gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size + 1)
            y_index = np.random.randint(0, w - crop_size + 1)
            processed_data[i, :, :, :] = train_data[index[i]][
                x_index : x_index + crop_size, y_index : y_index + crop_size, :
            ]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
    return gt_batch


def augment_1(x):
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))
    for j in range(vFlip):
        x = torch.flip(x, dims=(2,))
    for j in range(hFlip):
        x = torch.flip(x, dims=(1,))
    return x


def augment_2(generate_gt):
    c, h, w = (generate_gt.shape[1], 256, 256)
    divid_point_h = 128
    divid_point_w = 128
    output_img = generate_gt.new_zeros(c, h, w)
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img
