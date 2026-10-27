"""CIFAR-10 Dataset Loader for CNN Image Classifier.

Applied Robotics and AI Project (ARAIP)
University of Hertfordshire / PSB Academy
Student: Seenivasan M

This module provides utilities to download, load, and inspect the CIFAR-10
dataset using PyTorch and torchvision. Downloaded files are stored in the
'data/' directory, which is excluded from version control via .gitignore.
"""

import os
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Repository root and default data directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# CIFAR-10 class labels
CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def get_default_transform() -> transforms.Compose:
    """Return the default transform converting PIL images to PyTorch tensors."""
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def get_cifar10_datasets(
    data_dir: str = DEFAULT_DATA_DIR,
    download: bool = True,
    train_transform: Optional[transforms.Compose] = None,
    test_transform: Optional[transforms.Compose] = None,
) -> Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    """Load CIFAR-10 training and testing datasets.

    Args:
        data_dir: Directory where CIFAR-10 will be downloaded and stored.
                  Defaults to 'data/' in the project root (ignored by .gitignore).
        download: If True, downloads the dataset from the internet if not already present.
        train_transform: Optional torchvision transform for training data.
        test_transform: Optional torchvision transform for test data.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    os.makedirs(data_dir, exist_ok=True)

    if train_transform is None:
        train_transform = get_default_transform()
    if test_transform is None:
        test_transform = get_default_transform()

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=test_transform,
    )

    return train_dataset, test_dataset


def get_cifar10_loaders(
    data_dir: str = DEFAULT_DATA_DIR,
    batch_size: int = 64,
    num_workers: int = 0,
    download: bool = True,
    train_transform: Optional[transforms.Compose] = None,
    test_transform: Optional[transforms.Compose] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for CIFAR-10 train and test sets.

    Args:
        data_dir: Path to directory for storing dataset. Defaults to 'data/'.
        batch_size: Number of samples per batch. Defaults to 64.
        num_workers: Subprocesses for data loading. Defaults to 0 (safest for Windows).
        download: Whether to download if dataset is missing.
        train_transform: Optional transforms for training set.
        test_transform: Optional transforms for test set.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    train_dataset, test_dataset = get_cifar10_datasets(
        data_dir=data_dir,
        download=download,
        train_transform=train_transform,
        test_transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader


def display_sample_images(
    dataset: torchvision.datasets.CIFAR10,
    num_samples: int = 5,
    save_path: Optional[str] = None,
) -> None:
    """Display or save a row of sample images from the dataset with their class labels.

    Args:
        dataset: A CIFAR10 dataset instance.
        num_samples: Number of sample images to display.
        save_path: If specified, save figure to this file path instead of showing.
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(2.5 * num_samples, 2.5))
    if num_samples == 1:
        axes = [axes]

    for idx in range(num_samples):
        image, label = dataset[idx]
        # PyTorch tensors are (C, H, W); matplotlib expects (H, W, C)
        if isinstance(image, torch.Tensor):
            img_np = image.permute(1, 2, 0).numpy()
        else:
            img_np = image

        axes[idx].imshow(img_np)
        axes[idx].set_title(f"{CLASSES[label]} ({label})", fontsize=10)
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Sample images saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("CIFAR-10 Dataset Loader")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DEFAULT_DATA_DIR} (respects .gitignore: data/)")
    print("=" * 60)

    try:
        print("\nLoading CIFAR-10 datasets...")
        train_set, test_set = get_cifar10_datasets(download=True)

        print(f"Training samples: {len(train_set):,}")
        print(f"Testing samples:  {len(test_set):,}")
        print(f"Classes ({len(CLASSES)}): {', '.join(CLASSES)}")

        sample_img, sample_lbl = train_set[0]
        print(f"\nSample image tensor shape: {sample_img.shape} (Channels, Height, Width)")
        print(f"Sample image pixel range: [{sample_img.min():.2f}, {sample_img.max():.2f}]")
        print(f"Sample label: {sample_lbl} ({CLASSES[sample_lbl]})")

        train_ldr, test_ldr = get_cifar10_loaders(batch_size=64)
        images, labels = next(iter(train_ldr))
        print(f"\nDataLoader batch shape: {images.shape}")
        print(f"DataLoader labels batch shape: {labels.shape}")

        # Save sample image visualization
        sample_preview_path = os.path.join(PROJECT_ROOT, "docs", "screenshots", "cifar10-samples.png")
        display_sample_images(train_set, num_samples=5, save_path=sample_preview_path)
        print("\nDataset loader initialized successfully.")
    except Exception as exc:
        print(f"\nNotice: Dataset download or loading encountered an issue: {exc}")
        print("When connected to the internet, run 'python src/load_data.py' to download CIFAR-10 into data/.")
