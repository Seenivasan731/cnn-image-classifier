"""Baseline CNN Architecture and Training Loop for CIFAR-10.

Applied Robotics and AI Project (ARAIP)
University of Hertfordshire / PSB Academy
Student: Seenivasan M

This module defines:
  1. BaselineCNN: A PyTorch convolutional neural network tailored for 32x32 CIFAR-10 images.
  2. Training and evaluation routines with loss and accuracy tracking.
  3. Integration with data augmentation (RandomCrop, RandomHorizontalFlip) on the training pipeline.
  4. Model checkpoint saving and loading (saving weights to models/*.pth,
     which is ignored by Git per .gitignore).
"""

import os
import sys
import time
import argparse
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import (
    DEFAULT_DATA_DIR,
    CLASSES,
    get_preprocessed_loaders,
)

# Default path for model checkpoints (ignored by .gitignore: *.pth)
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "baseline_cnn.pth")


class BaselineCNN(nn.Module):
    """Convolutional Neural Network for CIFAR-10 image classification.

    Architecture:
      - Block 1: Conv2d(3 -> 32), BatchNorm, ReLU, MaxPool2d (32x32 -> 16x16)
      - Block 2: Conv2d(32 -> 64), BatchNorm, ReLU, MaxPool2d (16x16 -> 8x8)
      - Block 3: Conv2d(64 -> 128), BatchNorm, ReLU, MaxPool2d (8x8 -> 4x4)
      - Fully Connected: Linear(128*4*4 -> 256), ReLU, Dropout(0.5), Linear(256 -> 10)
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(BaselineCNN, self).__init__()

        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor and classifier."""
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_device() -> torch.device:
    """Return torch.device('cuda' if available else 'cpu')."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train the model for one epoch over the dataset.

    Returns:
        Tuple of (average_loss, accuracy_percentage).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = (100.0 * correct / total) if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on validation or test dataset.

    Returns:
        Tuple of (average_loss, accuracy_percentage).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = (100.0 * correct / total) if total > 0 else 0.0
    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    save_path: str = DEFAULT_MODEL_PATH,
    epoch: Optional[int] = None,
    val_acc: Optional[float] = None,
    optimizer: Optional[optim.Optimizer] = None,
) -> None:
    """Save model checkpoint to disk (.pth format, ignored by .gitignore)."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved: {save_path} (ignored by .gitignore: *.pth)")


def load_checkpoint(
    model: nn.Module,
    load_path: str = DEFAULT_MODEL_PATH,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, dict]:
    """Load model checkpoint weights from disk."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found at: {load_path}")

    checkpoint = torch.load(load_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    return model, checkpoint


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    num_epochs: int = 10,
    device: Optional[torch.device] = None,
    save_path: str = DEFAULT_MODEL_PATH,
) -> Dict[str, List[float]]:
    """Complete training loop tracking loss, accuracy, and saving the best model.

    Returns:
        History dictionary with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'.
    """
    if device is None:
        device = get_device()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    print(f"\nStarting training on device: {device} for {num_epochs} epochs...")
    print(f"Target checkpoint: {save_path}")

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        epoch_str = f"Epoch [{epoch:02d}/{num_epochs:02d}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            epoch_str += f" | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"

            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, save_path=save_path, epoch=epoch, val_acc=val_acc, optimizer=optimizer)
                epoch_str += " *"
        else:
            # If no validation set, save current epoch weights
            save_checkpoint(model, save_path=save_path, epoch=epoch, optimizer=optimizer)

        elapsed = time.time() - epoch_start
        epoch_str += f" ({elapsed:.1f}s)"
        print(epoch_str)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s. Best Validation Accuracy: {best_val_acc:.2f}%")
    return history


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train baseline CNN on CIFAR-10 with data augmentation")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DataLoaders (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer (default: 0.001)")
    parser.add_argument("--save-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to save model weights (.pth)")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable training data augmentation (RandomCrop, RandomHorizontalFlip)")
    parser.add_argument("--dry-run", action="store_true", help="Run model architecture verification without full training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    use_augmentation = not args.no_augmentation

    print("=" * 65)
    print("CIFAR-10 CNN Model & Training Loop")
    print(f"Project root:         {PROJECT_ROOT}")
    print(f"Active device:        {device}")
    print(f"Data augmentation:    {'ENABLED (RandomCrop, RandomHorizontalFlip)' if use_augmentation else 'DISABLED'}")
    print(f"Checkpoint save path: {args.save_path} (.pth ignored by .gitignore)")
    print("=" * 65)

    # Initialize model
    model = BaselineCNN(num_classes=len(CLASSES), dropout_rate=0.5).to(device)
    param_count = count_parameters(model)
    print(f"\nModel: BaselineCNN")
    print(f"Total trainable parameters: {param_count:,}")

    # Verify model forward pass with dummy tensor
    dummy_input = torch.randn(2, 3, 32, 32, device=device)
    dummy_output = model(dummy_input)
    assert dummy_output.shape == (2, len(CLASSES)), f"Unexpected output shape: {dummy_output.shape}"
    print(f"Dummy forward pass test: PASSED (input shape: [2, 3, 32, 32] -> output logits: {tuple(dummy_output.shape)})")

    # If dry-run requested, save an initial checkpoint to verify .pth handling
    if args.dry_run:
        print("\n[Dry Run] Saving test checkpoint to verify .pth handling...")
        save_checkpoint(model, save_path=args.save_path, epoch=0, val_acc=0.0)
        print("Dry run complete.")
        sys.exit(0)

    # Attempt dataset loading and training
    try:
        print(f"\nLoading preprocessed CIFAR-10 DataLoaders (augmentation: {use_augmentation})...")
        train_loader, val_loader, test_loader = get_preprocessed_loaders(
            batch_size=args.batch_size,
            val_split=0.1,
            num_workers=0,
            download=True,
            use_augmentation=use_augmentation,
        )

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            device=device,
            save_path=args.save_path,
        )
    except Exception as exc:
        print(f"\nNotice: Dataset loading encountered an issue: {exc}")
        print("Saving initialized baseline model weights checkpoint...")
        save_checkpoint(model, save_path=args.save_path, epoch=0, val_acc=0.0)
        print(f"When connected to the internet, run 'python src/train.py --epochs 5' to train on CIFAR-10.")
