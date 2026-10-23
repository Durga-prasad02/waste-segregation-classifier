"""
Training Pipeline for Waste Segregation Image Classifier
Author: AGGIBADALA VENKATA DURGA NAGA PRASAD (23ME1A5402)
Guide: Ms. G. Hema Sri
Domain: Deep Learning & Computer Vision

Features:
- Image augmentation (Rotation, Flips, Color Jitter, Normalization)
- Custom CNN and MobileNetV2 Transfer Learning support
- Metrics: Cross-Entropy Loss, Top-1 Accuracy, Precision, Recall, F1
- Checkpoint persistence with metadata (classes, architecture, best accuracy)
- Evaluation confusion matrix & loss curve plotting
"""

import os
import sys
import tempfile

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from model import build_model, WASTE_CLASSES

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
IMAGE_SIZE = (224, 224)
MODEL_SAVE_PATH = "waste_classifier_model.pth"


def get_data_transforms():
    """Builds standard training augmentations and validation transforms tailored to waste images."""
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_model(
    data_dir: str = "dataset",
    model_type: str = "custom",
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    save_path: str = MODEL_SAVE_PATH
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Training Pipeline on device: {device}")
    print(f"📦 Model Architecture: {model_type.upper()}")

    train_transform, val_transform = get_data_transforms()
    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"

    if not train_dir.exists() or not any(train_dir.iterdir()):
        print("⚠️ Training data not found. Auto-bootstrapping initial dataset...")
        from dataset_downloader import setup_dataset_structure
        setup_dataset_structure()

    # Load datasets
    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    print(f"📊 Detected {len(class_names)} Classes: {class_names}")
    print(f"📊 Training Samples: {len(train_dataset)} | Validation Samples: {len(val_dataset)}")

    # Instantiate model
    model = build_model(model_name=model_type, num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_train += inputs.size(0)

        scheduler.step()

        epoch_train_loss = running_loss / total_train if total_train > 0 else 0
        epoch_train_acc = (running_corrects / total_train) * 100 if total_train > 0 else 0

        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data).item()
                total_val += inputs.size(0)

        epoch_val_loss = val_running_loss / total_val if total_val > 0 else 0
        epoch_val_acc = (val_running_corrects / total_val) * 100 if total_val > 0 else 0

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")

        # Save checkpoint if best validation accuracy achieved
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_type": model_type,
                "class_names": class_names,
                "best_acc": best_val_acc,
                "history": history
            }
            torch.save(checkpoint, save_path)

    elapsed_time = time.time() - start_time
    print(f"\n🎉 Training finished in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"💾 Checkpoint saved to: {Path(save_path).resolve()}")

    # Generate and save metric plots
    plot_training_history(history, save_plot_path="training_metrics.png")
    return model, history


def plot_training_history(history: dict, save_plot_path: str = "training_metrics.png"):
    """Saves Loss and Accuracy curve figure for project documentation and presentation."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "b-o", label="Training Loss")
    ax1.plot(epochs, history["val_loss"], "r--s", label="Validation Loss")
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Accuracy curve
    ax2.plot(epochs, history["train_acc"], "b-o", label="Training Accuracy (%)")
    ax2.plot(epochs, history["val_acc"], "g--s", label="Validation Accuracy (%)")
    ax2.set_title("Classification Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=300)
    plt.close()
    print(f"📈 Saved performance curves to: {Path(save_plot_path).resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Waste Segregation Classifier")
    parser.add_argument("--model", type=str, default="custom", choices=["custom", "mobilenet"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_model(model_type=args.model, num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
