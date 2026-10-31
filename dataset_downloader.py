"""
Dataset Manager for Waste Segregation Classifier
Author: AGGIBADALA VENKATA DURGA NAGA PRASAD (23ME1A5402)
Guide: Ms. G. Hema Sri
Domain: Deep Learning & Computer Vision
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
from PIL import Image
import torchvision.transforms as T

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def setup_dataset_structure(
    base_dir: str = "dataset",
    samples_dir: str = "samples",
    train_aug_count: int = 40,
    val_aug_count: int = 10
):
    """
    Sets up the full train/val dataset directory structure and bootstraps augmented
    real-world samples from the demonstration gallery.
    """
    base_path = Path(base_dir)
    train_path = base_path / "train"
    val_path = base_path / "val"
    samples_path = Path(samples_dir)

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    augmenter = T.Compose([
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])

    print(f"📁 Initializing Dataset in: {base_path.resolve()}")
    for cls in CLASSES:
        cls_train_dir = train_path / cls
        cls_val_dir = val_path / cls
        cls_train_dir.mkdir(parents=True, exist_ok=True)
        cls_val_dir.mkdir(parents=True, exist_ok=True)

        sample_file = samples_path / f"demo_{cls}.jpg"
        if not sample_file.exists():
            print(f"⚠️ Missing prototype image: {sample_file}")
            continue

        base_img = Image.open(sample_file).convert("RGB")
        
        # Save canonical image
        base_img.resize((224, 224)).save(cls_train_dir / f"{cls}_000.jpg", quality=95)
        base_img.resize((224, 224)).save(cls_val_dir / f"{cls}_val_000.jpg", quality=95)

        # Generate training variations
        for i in range(1, train_aug_count):
            aug_img = augmenter(base_img)
            aug_img.save(cls_train_dir / f"{cls}_{i:03d}.jpg", quality=92)

        # Generate validation variations
        for i in range(1, val_aug_count):
            aug_img = augmenter(base_img)
            aug_img.save(cls_val_dir / f"{cls}_val_{i:03d}.jpg", quality=92)

    print(f"✅ Generated training ({train_aug_count * len(CLASSES)}) and validation ({val_aug_count * len(CLASSES)}) samples across all {len(CLASSES)} waste categories.")

if __name__ == "__main__":
    setup_dataset_structure()
