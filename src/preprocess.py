"""CIFAR-10 Data Preprocessing and Train/Test Splitting.

Applied Robotics and AI Project (ARAIP)
University of Hertfordshire / PSB Academy
Student: Seenivasan M

This module provides data preprocessing utilities for the CIFAR-10 dataset,
including:
  1. Per-channel image normalization (standard CIFAR-10 mean and std).
  2. Inverse normalization (denormalization) for image visualization.
  3. Deterministic train/validation/test splitting.
  4. Preprocessed PyTorch DataLoader generation.
"""

import os
import sys
from typing import Tuple, Optional, Union
import torch
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.transforms as transforms

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.load_data import (
    DEFAULT_DATA_DIR,
    CLASSES,
    get_cifar10_datasets,
)

# Standard empirical mean and standard deviation for CIFAR-10 (RGB channels)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_normalization_transform(
    mean: Tuple[float, float, float] = CIFAR10_MEAN,
    std: Tuple[float, float, float] = CIFAR10_STD,
) -> transforms.Compose:
    """Return a torchvision transform that converts PIL images to tensors and normalizes them.

    Args:
        mean: Channel-wise means for R, G, B channels.
        std: Channel-wise standard deviations for R, G, B channels.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = CIFAR10_MEAN,
    std: Tuple[float, float, float] = CIFAR10_STD,
) -> torch.Tensor:
    """Reverse per-channel normalization on an image tensor for display or inspection.

    Args:
        tensor: Tensor of shape (C, H, W) or (B, C, H, W).
        mean: Normalization means used during preprocessing.
        std: Normalization standard deviations used during preprocessing.

    Returns:
        Denormalized tensor with pixel values clamped to [0.0, 1.0].
    """
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_tensor = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    if tensor.ndim == 3:
        # Shape: (C, H, W)
        mean_tensor = mean_tensor.view(-1, 1, 1)
        std_tensor = std_tensor.view(-1, 1, 1)
    elif tensor.ndim == 4:
        # Shape: (B, C, H, W)
        mean_tensor = mean_tensor.view(1, -1, 1, 1)
        std_tensor = std_tensor.view(1, -1, 1, 1)

    denorm = tensor * std_tensor + mean_tensor
    return torch.clamp(denorm, 0.0, 1.0)


def get_preprocessed_datasets(
    data_dir: str = DEFAULT_DATA_DIR,
    val_split: float = 0.1,
    random_seed: int = 42,
    download: bool = True,
    mean: Tuple[float, float, float] = CIFAR10_MEAN,
    std: Tuple[float, float, float] = CIFAR10_STD,
) -> Tuple[Union[Subset, torch.utils.data.Dataset], Optional[Subset], torch.utils.data.Dataset]:
    """Load normalized CIFAR-10 and split the training set into train/validation sets.

    Args:
        data_dir: Root directory for dataset storage (ignored by .gitignore).
        val_split: Fraction of the training set (0.0 to 1.0) to reserve for validation.
                   Default is 0.1 (5,000 validation images from 50,000 training images).
        random_seed: Random seed for reproducible splitting.
        download: Download dataset if not locally present.
        mean: Per-channel normalization mean.
        std: Per-channel normalization std.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset). If val_split is 0,
        val_dataset will be None.
    """
    transform = get_normalization_transform(mean=mean, std=std)

    full_train_dataset, test_dataset = get_cifar10_datasets(
        data_dir=data_dir,
        download=download,
        train_transform=transform,
        test_transform=transform,
    )

    if val_split > 0.0:
        total_train = len(full_train_dataset)
        val_size = int(total_train * val_split)
        train_size = total_train - val_size

        generator = torch.Generator().manual_seed(random_seed)
        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=generator,
        )
        return train_dataset, val_dataset, test_dataset

    return full_train_dataset, None, test_dataset


def get_preprocessed_loaders(
    data_dir: str = DEFAULT_DATA_DIR,
    batch_size: int = 64,
    val_split: float = 0.1,
    random_seed: int = 42,
    num_workers: int = 0,
    download: bool = True,
    mean: Tuple[float, float, float] = CIFAR10_MEAN,
    std: Tuple[float, float, float] = CIFAR10_STD,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Create normalized DataLoaders for train, validation, and test sets.

    Args:
        data_dir: Path to directory for dataset. Defaults to 'data/'.
        batch_size: Samples per batch. Defaults to 64.
        val_split: Fraction of train dataset used for validation.
        random_seed: Seed for reproducible train/val split.
        num_workers: Subprocesses for data loading (0 recommended on Windows).
        download: Whether to download if dataset is missing.
        mean: Per-channel mean for normalization.
        std: Per-channel std for normalization.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_dataset, val_dataset, test_dataset = get_preprocessed_datasets(
        data_dir=data_dir,
        val_split=val_split,
        random_seed=random_seed,
        download=download,
        mean=mean,
        std=std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def verify_batch_statistics(images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute channel-wise mean and std of a batch of image tensors to verify normalization."""
    # images shape: (B, C, H, W) -> permute to (C, B, H, W) and compute over remaining dimensions
    mean = images.mean(dim=[0, 2, 3])
    std = images.std(dim=[0, 2, 3])
    return mean, std


if __name__ == "__main__":
    print("=" * 65)
    print("CIFAR-10 Data Preprocessing (Normalization & Train/Test Split)")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Target data directory: {DEFAULT_DATA_DIR}")
    print(f"Normalization Mean (RGB): {CIFAR10_MEAN}")
    print(f"Normalization Std  (RGB): {CIFAR10_STD}")
    print("=" * 65)

    try:
        print("\nLoading preprocessed datasets with 90/10 train-validation split...")
        train_set, val_set, test_set = get_preprocessed_datasets(
            val_split=0.1,
            random_seed=42,
            download=True,
        )

        val_count = len(val_set) if val_set is not None else 0
        print(f"\nDataset split counts:")
        print(f"  - Training samples:   {len(train_set):,}")
        print(f"  - Validation samples: {val_count:,}")
        print(f"  - Testing samples:    {len(test_set):,}")
        print(f"  - Total samples:      {len(train_set) + val_count + len(test_set):,}")

        train_ldr, val_ldr, test_ldr = get_preprocessed_loaders(
            batch_size=64,
            val_split=0.1,
            num_workers=0,
            download=True,
        )

        batch_imgs, batch_lbls = next(iter(train_ldr))
        ch_mean, ch_std = verify_batch_statistics(batch_imgs)

        print(f"\nDataLoader Batch Verification:")
        print(f"  - Batch image tensor shape: {batch_imgs.shape} (B, C, H, W)")
        print(f"  - Batch labels tensor shape: {batch_lbls.shape}")
        print(f"  - Normalized pixel range:   [{batch_imgs.min():.2f}, {batch_imgs.max():.2f}]")
        print(f"  - Batch channel mean (RGB): [{ch_mean[0]:.3f}, {ch_mean[1]:.3f}, {ch_mean[2]:.3f}] (centered near 0.0)")
        print(f"  - Batch channel std  (RGB): [{ch_std[0]:.3f}, {ch_std[1]:.3f}, {ch_std[2]:.3f}] (scaled near 1.0)")

        # Verify denormalization
        denorm_batch = denormalize(batch_imgs)
        print(f"\nDenormalization Verification:")
        print(f"  - Restored pixel range:     [{denorm_batch.min():.2f}, {denorm_batch.max():.2f}] (within [0, 1])")

        print("\nData preprocessing module initialized and verified successfully.")

    except Exception as exc:
        print(f"\nNotice: Dataset download or loading encountered an issue: {exc}")
        print("When connected to the internet, run 'python src/preprocess.py' to download and preprocess.")
