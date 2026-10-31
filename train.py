"""
Training Pipeline for Waste Segregation Image Classifier
Author: AGGIBADALA VENKATA DURGA NAGA PRASAD (23ME1A5402)
Guide: Ms. G. Hema Sri
Domain: Deep Learning & Computer Vision
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

from model import build_model, WASTE_CLASSES

BATCH_SIZE = 16
NUM_EPOCHS = 8
LEARNING_RATE = 1e-3
IMAGE_SIZE = (224, 224)
MODEL_SAVE_PATH = "waste_classifier_model.pth"


def get_data_transforms():
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
    save_path: str = MODEL_SAVE_PATH,
    resume: bool = False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Training Pipeline on device: {device}")
    print(f"📦 Model Architecture: {model_type.upper()}")

    train_transform, val_transform = get_data_transforms()
    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"

    if not train_dir.exists() or not any(train_dir.iterdir()):
        print("⚠️ Training data not found. Auto-bootstrapping dataset from samples...")
        from dataset_downloader import setup_dataset_structure
        setup_dataset_structure()

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    print(f"📊 Detected {len(class_names)} Classes: {class_names}")
    print(f"📊 Training Samples: {len(train_dataset)} | Validation Samples: {len(val_dataset)}")

    model = build_model(model_name=model_type, num_classes=len(class_names), pretrained=True)
    if resume and Path(save_path).exists():
        print(f"🔄 Resuming weights from existing checkpoint: {save_path}")
        try:
            ckpt = torch.load(save_path, map_location=device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            else:
                model.load_state_dict(ckpt)
        except Exception as e:
            print(f"⚠️ Could not load existing checkpoint: {e}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

        scheduler.step()

        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = (running_corrects / len(train_dataset)) * 100.0

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                v_loss = criterion(val_outputs, val_labels)

                _, v_preds = torch.max(val_outputs, 1)
                val_running_loss += v_loss.item() * val_inputs.size(0)
                val_running_corrects += torch.sum(v_preds == val_labels.data).item()

        epoch_val_loss = val_running_loss / len(val_dataset)
        epoch_val_acc = (val_running_corrects / len(val_dataset)) * 100.0

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_architecture": model_type,
                "classes": class_names,
                "best_acc": best_val_acc,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, save_path)

    elapsed_time = time.time() - start_time
    print(f"\n🎉 Training finished in {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"💾 Checkpoint saved to: {Path(save_path).resolve()}")

    # Plot metrics
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    plt.plot(range(1, num_epochs + 1), history["val_loss"], label="Val Loss", color="crimson", lw=2, linestyle="--")
    plt.title("Cross-Entropy Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), history["train_acc"], label="Train Acc", color="royalblue", lw=2)
    plt.plot(range(1, num_epochs + 1), history["val_acc"], label="Val Acc", color="forestgreen", lw=2, linestyle="--")
    plt.title("Accuracy (%)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    metrics_path = Path("training_metrics.png")
    plt.tight_layout()
    plt.savefig(metrics_path, dpi=200)
    plt.close()
    print(f"📈 Saved performance curves to: {metrics_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN Classifier for Waste Segregation")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Directory with train and val folders")
    parser.add_argument("--model", type=str, default="custom", choices=["custom", "mobilenet_v2"], help="Model type")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--save_path", type=str, default=MODEL_SAVE_PATH, help="Path to save weights")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        model_type=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path,
        resume=args.resume
    )
