"""CIFAR-10 Data Preprocessing, Augmentation, and Train/Test Splitting.

Applied Robotics and AI Project (ARAIP)
University of Hertfordshire / PSB Academy
Student: Seenivasan M

This module provides data preprocessing and augmentation utilities for the
CIFAR-10 dataset, including:
  1. Data augmentation for training images:
       - RandomCrop(32, padding=4, padding_mode='reflect')
       - RandomHorizontalFlip(p=0.5)
  2. Per-channel image normalization (standard CIFAR-10 mean and std).
  3. Inverse normalization (denormalization) for image visualization.
  4. Deterministic train/validation/test splitting (training with augmentation,
     validation/test with clean deterministic normalization).
  5. Preprocessed PyTorch DataLoader generation.
"""

import os
import sys
from typing import Tuple, Optional, Union
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
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


def get_augmentation_transform(
    mean: Tuple[float, float, float] = CIFAR10_MEAN,
    std: Tuple[float, float, float] = CIFAR10_STD,
    crop_padding: int = 4,
    flip_prob: float = 0.5,
) -> transforms.Compose:
    """Return a torchvision transform pipeline that applies data augmentation and normalization.

    Augmentations applied:
      - RandomCrop(32, padding=crop_padding, padding_mode='reflect'): Shifts the image
        by up to crop_padding pixels in any direction to introduce translational invariance.
      - RandomHorizontalFlip(p=flip_prob): Randomly mirrors the image horizontally to
        teach the model reflectional symmetry.
      - ToTensor(): Converts PIL image to PyTorch FloatTensor in range [0.0, 1.0].
      - Normalize(mean, std): Standardizes pixel intensities across each RGB channel.

    Args:
        mean: Channel-wise means for R, G, B channels.
        std: Channel-wise standard deviations for R, G, B channels.
        crop_padding: Pixels to pad on each border before random 32x32 crop.
        flip_prob: Probability of horizontal flip.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=crop_padding, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(p=flip_prob),
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
    use_augmentation: bool = True,
) -> Tuple[Union[Subset, torch.utils.data.Dataset], Optional[Subset], torch.utils.data.Dataset]:
    """Load preprocessed CIFAR-10 with data augmentation on training and deterministic eval on val/test.

    Args:
        data_dir: Root directory for dataset storage (ignored by .gitignore).
        val_split: Fraction of the training set (0.0 to 1.0) to reserve for validation.
                   Default is 0.1 (5,000 validation images from 50,000 training images).
        random_seed: Random seed for reproducible splitting.
        download: Download dataset if not locally present.
        mean: Per-channel normalization mean.
        std: Per-channel normalization std.
        use_augmentation: If True, applies RandomCrop and RandomHorizontalFlip to training data.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset). If val_split is 0,
        val_dataset will be None.
    """
    train_transform = get_augmentation_transform(mean=mean, std=std) if use_augmentation else get_normalization_transform(mean=mean, std=std)
    eval_transform = get_normalization_transform(mean=mean, std=std)

    # Full train dataset with training transform (augmented)
    full_train_dataset, _ = get_cifar10_datasets(
        data_dir=data_dir,
        download=download,
        train_transform=train_transform,
        test_transform=eval_transform,
    )

    # Clean test dataset (unaugmented, only normalized)
    _, test_dataset = get_cifar10_datasets(
        data_dir=data_dir,
        download=download,
        train_transform=eval_transform,
        test_transform=eval_transform,
    )

    if val_split > 0.0:
        total_train = len(full_train_dataset)
        val_size = int(total_train * val_split)
        train_size = total_train - val_size

        # Create validation dataset with eval_transform (no random augmentation)
        full_val_dataset, _ = get_cifar10_datasets(
            data_dir=data_dir,
            download=download,
            train_transform=eval_transform,
            test_transform=eval_transform,
        )

        generator = torch.Generator().manual_seed(random_seed)
        shuffled_indices = torch.randperm(total_train, generator=generator).tolist()
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:]

        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_val_dataset, val_indices)
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
    use_augmentation: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Create DataLoaders for train (with augmentation), validation, and test sets.

    Args:
        data_dir: Path to directory for dataset. Defaults to 'data/'.
        batch_size: Samples per batch. Defaults to 64.
        val_split: Fraction of train dataset used for validation.
        random_seed: Seed for reproducible train/val split.
        num_workers: Subprocesses for data loading (0 recommended on Windows).
        download: Whether to download if dataset is missing.
        mean: Per-channel mean for normalization.
        std: Per-channel std for normalization.
        use_augmentation: Whether to apply data augmentation to training data.

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
        use_augmentation=use_augmentation,
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
    # images shape: (B, C, H, W) -> compute over batch and spatial dimensions
    mean = images.mean(dim=[0, 2, 3])
    std = images.std(dim=[0, 2, 3])
    return mean, std


if __name__ == "__main__":
    print("=" * 65)
    print("CIFAR-10 Data Preprocessing & Data Augmentation Pipeline")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Target data directory: {DEFAULT_DATA_DIR}")
    print(f"Normalization Mean (RGB): {CIFAR10_MEAN}")
    print(f"Normalization Std  (RGB): {CIFAR10_STD}")
    print(f"Augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip(p=0.5)")
    print("=" * 65)

    try:
        print("\nLoading preprocessed datasets with data augmentation enabled...")
        train_set, val_set, test_set = get_preprocessed_datasets(
            val_split=0.1,
            random_seed=42,
            download=True,
            use_augmentation=True,
        )

        val_count = len(val_set) if val_set is not None else 0
        print(f"\nDataset split counts:")
        print(f"  - Training samples (augmented):   {len(train_set):,}")
        print(f"  - Validation samples (eval-only): {val_count:,}")
        print(f"  - Testing samples (eval-only):    {len(test_set):,}")
        print(f"  - Total samples:                  {len(train_set) + val_count + len(test_set):,}")

        train_ldr, val_ldr, test_ldr = get_preprocessed_loaders(
            batch_size=64,
            val_split=0.1,
            num_workers=0,
            download=True,
            use_augmentation=True,
        )

        batch_imgs, batch_lbls = next(iter(train_ldr))
        ch_mean, ch_std = verify_batch_statistics(batch_imgs)

        print(f"\nAugmented DataLoader Batch Verification:")
        print(f"  - Batch image tensor shape: {batch_imgs.shape} (B, C, H, W)")
        print(f"  - Batch labels tensor shape: {batch_lbls.shape}")
        print(f"  - Normalized pixel range:   [{batch_imgs.min():.2f}, {batch_imgs.max():.2f}]")
        print(f"  - Batch channel mean (RGB): [{ch_mean[0]:.3f}, {ch_mean[1]:.3f}, {ch_mean[2]:.3f}] (centered near 0.0)")
        print(f"  - Batch channel std  (RGB): [{ch_std[0]:.3f}, {ch_std[1]:.3f}, {ch_std[2]:.3f}] (scaled near 1.0)")

        # Verify denormalization
        denorm_batch = denormalize(batch_imgs)
        print(f"\nDenormalization Verification:")
        print(f"  - Restored pixel range:     [{denorm_batch.min():.2f}, {denorm_batch.max():.2f}] (within [0, 1])")

        print("\nData preprocessing & augmentation module initialized and verified successfully.")

    except Exception as exc:
        print(f"\nNotice: Dataset download or loading encountered an issue: {exc}")
        print("When connected to the internet, run 'python src/preprocess.py' to download and test augmentation.")
