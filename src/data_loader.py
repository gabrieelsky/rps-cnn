import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

from src.config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    HFLIP_PROB,
    VFLIP_PROB,
    ROTATION_DEGREES,
    RANDOM_RESIZED_CROP_SCALE,
    RANDOM_RESIZED_CROP_RATIO,
    COLOR_JITTER_BRIGHTNESS,
    COLOR_JITTER_CONTRAST,
    COLOR_JITTER_SATURATION,
)

class RPSDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]

        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
def get_class_mapping(data_dir):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    return {cls_name: idx for idx, cls_name in enumerate(classes)}


def _as_rgb_stats(mean_value, std_value):
    """Return stats as 3-element lists expected by torchvision Normalize."""
    mean = [float(mean_value)] * 3
    std = [max(float(std_value), 1e-6)] * 3
    return mean, std


def get_train_transform(img_size=(IMG_HEIGHT, IMG_WIDTH), norm_mean=None, norm_std=None):
    if norm_mean is None or norm_std is None:
        raise ValueError("get_train_transform requires explicit norm_mean/norm_std values.")

    return v2.Compose([
        v2.Resize(img_size),
        # Crop/zoom randomization mitigates fixed edge artifacts and positional shortcuts.
        v2.RandomResizedCrop(
            size=img_size,
            scale=RANDOM_RESIZED_CROP_SCALE,
            ratio=RANDOM_RESIZED_CROP_RATIO,
            antialias=True,
        ),
        v2.RandomHorizontalFlip(p=HFLIP_PROB),
        v2.RandomVerticalFlip(p=VFLIP_PROB),
        v2.RandomRotation(degrees=ROTATION_DEGREES),
        v2.ColorJitter(
            brightness=COLOR_JITTER_BRIGHTNESS,
            contrast=COLOR_JITTER_CONTRAST,
            saturation=COLOR_JITTER_SATURATION,
        ),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=norm_mean, std=norm_std),
    ])


def get_eval_transform(img_size=(IMG_HEIGHT, IMG_WIDTH), norm_mean=None, norm_std=None):
    if norm_mean is None or norm_std is None:
        raise ValueError("get_eval_transform requires explicit norm_mean/norm_std values.")

    return v2.Compose([
        v2.Resize(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=norm_mean, std=norm_std),
    ])

def create_dataloaders(data_dir, batch_size=32, img_size=(IMG_HEIGHT, IMG_WIDTH), seed=42):
    file_paths = []
    labels = []

    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        for file_name in os.listdir(cls_dir):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_paths.append(os.path.join(cls_dir, file_name))
                labels.append(class_to_idx[cls_name])

    # Perform stratified data splitting
    # 70% Train, 30% Temporary (Val + Test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths, labels, test_size=0.30, stratify=labels, random_state=seed
    )
    
    # Split into Validation (15%) and Test (15%)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=seed
    )

    # Methodologically correct: compute normalization stats using only training split.
    scalar_mean, scalar_std = compute_global_mean_std(train_paths, img_size)

    norm_mean, norm_std = _as_rgb_stats(scalar_mean, scalar_std)

    train_transform = get_train_transform(img_size, norm_mean=norm_mean, norm_std=norm_std)
    val_test_transform = get_eval_transform(img_size, norm_mean=norm_mean, norm_std=norm_std)

    train_dataset = RPSDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = RPSDataset(val_paths, val_labels, transform=val_test_transform)
    test_dataset = RPSDataset(test_paths, test_labels, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_to_idx


def compute_global_mean_std(file_paths, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Compute global mean/std in [0,1] over RGB pixels from the provided file list."""
    pixel_count = 0
    pixel_sum = 0.0
    pixel_sq_sum = 0.0

    for p in file_paths:
        with Image.open(p) as img:
            img = img.convert('RGB').resize((img_size[1], img_size[0]))
            arr = np.asarray(img, dtype=np.float32) / 255.0  # H,W,C
            flat = arr.reshape(-1)
            pixel_sum += float(flat.sum())
            pixel_sq_sum += float((flat ** 2).sum())
            pixel_count += flat.size

    if pixel_count == 0:
        # Rare fallback for empty datasets.
        return 0.5, 0.25

    mean = pixel_sum / pixel_count
    var = max((pixel_sq_sum / pixel_count) - (mean ** 2), 1e-8)
    std = float(np.sqrt(var))
    return float(mean), std


def compute_mean_std(file_paths, img_size=(200,300)):
    """Compute per-channel mean and std (in [0,1]) for a list of image file paths.

    Returns (mean, std) as numpy arrays of shape (3,).
    """
    from PIL import Image

    cnt = 0
    mean = np.zeros(3, dtype=np.float64)
    sq_mean = np.zeros(3, dtype=np.float64)

    for p in file_paths:
        with Image.open(p) as img:
            img = img.convert('RGB').resize((img_size[1], img_size[0]))
            arr = np.asarray(img, dtype=np.float32) / 255.0  # H,W,C
            # sum over H,W
            mean += arr.mean(axis=(0, 1))
            sq_mean += (arr ** 2).mean(axis=(0, 1))
            cnt += 1

    if cnt == 0:
        # fallback to ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return mean, std

    mean = mean / cnt
    sq_mean = sq_mean / cnt
    var = sq_mean - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)