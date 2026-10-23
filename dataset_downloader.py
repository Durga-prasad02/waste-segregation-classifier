"""
Dataset Manager & Generator for Waste Segregation Image Classifier
Author: AGGIBADALA VENKATA DURGA NAGA PRASAD (23ME1A5402)
Guide: Ms. G. Hema Sri

This module provides tools to:
1. Download standard benchmark datasets (TrashNet / Municipal Waste Dataset).
2. Generate structured train/val directory layouts.
3. Generate bootstrap / mock sample images using Pillow with distinct visual patterns
   for instant offline demonstration and testing.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import shutil
import urllib.request
import zipfile
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Color themes & shapes for synthetic demonstration generation
CLASS_PROPERTIES = {
    "cardboard": {"base_rgb": (180, 130, 80), "shape": "box", "accent": (140, 95, 50)},
    "glass": {"base_rgb": (80, 170, 120), "shape": "bottle", "accent": (130, 220, 180)},
    "metal": {"base_rgb": (195, 195, 205), "shape": "can", "accent": (140, 140, 150)},
    "paper": {"base_rgb": (245, 245, 240), "shape": "sheet", "accent": (100, 120, 150)},
    "plastic": {"base_rgb": (70, 150, 220), "shape": "bottle", "accent": (230, 240, 255)},
    "trash": {"base_rgb": (90, 80, 75), "shape": "organic", "accent": (60, 50, 45)},
}


def create_synthetic_waste_image(category: str, width: int = 224, height: int = 224) -> Image.Image:
    """Creates a visually distinguishable waste category sample image with realistic textures & artifacts."""
    props = CLASS_PROPERTIES.get(category, CLASS_PROPERTIES["trash"])
    
    # Background (ambient floor / bin lighting)
    bg_color = (
        random.randint(220, 245),
        random.randint(220, 245),
        random.randint(220, 245)
    )
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Object bounding box
    margin_x = random.randint(30, 50)
    margin_y = random.randint(30, 50)
    x1, y1 = margin_x, margin_y
    x2, y2 = width - margin_x, height - margin_y
    
    base_color = props["base_rgb"]
    jitter_color = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in base_color)
    accent_color = props["accent"]
    shape = props["shape"]

    if shape == "box":
        # Draw 3D-ish corrugated cardboard box
        draw.rectangle([x1, y1, x2, y2], fill=jitter_color, outline=accent_color, width=3)
        # Flaps and seams
        draw.line([(x1, y1 + (y2 - y1) // 3), (x2, y1 + (y2 - y1) // 3)], fill=accent_color, width=2)
        draw.line([(x1 + (x2 - x1) // 2, y1), (x1 + (x2 - x1) // 2, y2)], fill=accent_color, width=2)
        # Packaging label
        draw.rectangle([x1 + 20, y1 + 50, x1 + 60, y1 + 80], fill=(240, 235, 220), outline=(100, 100, 100))
        
    elif shape == "bottle":
        # Draw bottle outline (neck and body)
        neck_w = (x2 - x1) // 3
        neck_left = x1 + neck_w
        neck_right = x2 - neck_w
        body_top = y1 + (y2 - y1) // 3
        
        # Neck
        draw.rectangle([neck_left, y1, neck_right, body_top], fill=jitter_color, outline=accent_color, width=2)
        # Body
        draw.rounded_rectangle([x1, body_top, x2, y2], radius=15, fill=jitter_color, outline=accent_color, width=2)
        # Cap
        draw.rectangle([neck_left - 2, y1 - 8, neck_right + 2, y1], fill=(220, 50, 50), outline=(150, 30, 30))
        # Gloss reflection highlights
        draw.line([(x1 + 15, body_top + 10), (x1 + 15, y2 - 20)], fill=(255, 255, 255), width=3)

    elif shape == "can":
        # Cylindrical beverage / food can
        draw.rounded_rectangle([x1 + 10, y1, x2 - 10, y2], radius=20, fill=jitter_color, outline=accent_color, width=3)
        # Ellipse top & bottom rims
        draw.ellipse([x1 + 10, y1 - 5, x2 - 10, y1 + 25], outline=accent_color, width=2)
        draw.ellipse([x1 + 10, y2 - 25, x2 - 10, y2 + 5], outline=accent_color, width=2)
        # Metallic band
        mid_y = (y1 + y2) // 2
        draw.rectangle([x1 + 10, mid_y - 20, x2 - 10, mid_y + 20], fill=accent_color)
        draw.text((x1 + 25, mid_y - 8), "ALU", fill=(255, 255, 255))

    elif shape == "sheet":
        # Creased paper / document
        points = [(x1, y1), (x2 - 20, y1), (x2, y1 + 25), (x2, y2), (x1 + 10, y2 - 5), (x1, y2)]
        draw.polygon(points, fill=jitter_color, outline=(180, 180, 180), width=2)
        # Dog-ear corner fold
        draw.polygon([(x2 - 20, y1), (x2 - 20, y1 + 25), (x2, y1 + 25)], fill=(210, 210, 210), outline=(150, 150, 150))
        # Text simulation lines
        for ly in range(y1 + 40, y2 - 20, 14):
            line_end = random.randint(x1 + 40, x2 - 20)
            draw.line([(x1 + 15, ly), (line_end, ly)], fill=(80, 80, 90), width=2)

    elif shape == "organic":
        # Organic food waste / crumpled non-recyclable item
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = (x2 - x1) // 2 - 10
        # Multi-vertex organic shape
        num_vertices = 12
        poly_points = []
        for i in range(num_vertices):
            angle = i * (2 * np.pi / num_vertices)
            r = radius + random.randint(-15, 15)
            px = center_x + int(r * np.cos(angle))
            py = center_y + int(r * np.sin(angle))
            poly_points.append((px, py))
        draw.polygon(poly_points, fill=jitter_color, outline=accent_color, width=2)
        # Texture spots
        for _ in range(15):
            sx = random.randint(x1 + 15, x2 - 15)
            sy = random.randint(y1 + 15, y2 - 15)
            draw.ellipse([sx, sy, sx + 6, sy + 6], fill=(50, 40, 30))

    # Add Category label tag at bottom
    draw.rectangle([width // 2 - 45, height - 25, width // 2 + 45, height - 5], fill=(30, 30, 30))
    draw.text((width // 2 - 35, height - 22), category.upper()[:8], fill=(255, 255, 255))
    
    # Slight blur to mimic natural camera focal softening
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def setup_dataset_structure(base_dir: str = "dataset", samples_per_class_train: int = 40, samples_per_class_val: int = 10):
    """
    Sets up the full train/val dataset directory structure and bootstraps sample images.
    """
    base_path = Path(base_dir)
    train_path = base_path / "train"
    val_path = base_path / "val"
    samples_path = Path("samples")

    samples_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Setting up dataset folders in: {base_path.resolve()}")
    for cls in CLASSES:
        (train_path / cls).mkdir(parents=True, exist_ok=True)
        (val_path / cls).mkdir(parents=True, exist_ok=True)
        
        # Populate training images
        for i in range(samples_per_class_train):
            img = create_synthetic_waste_image(cls)
            img.save(train_path / cls / f"{cls}_{i:03d}.jpg", quality=92)
            
        # Populate validation images
        for i in range(samples_per_class_val):
            img = create_synthetic_waste_image(cls)
            img.save(val_path / cls / f"{cls}_val_{i:03d}.jpg", quality=92)

        # Save high-res presentation sample for easy web UI testing
        sample_img = create_synthetic_waste_image(cls, width=300, height=300)
        sample_img.save(samples_path / f"demo_{cls}.jpg", quality=95)
        
    print(f"✅ Generated {samples_per_class_train * len(CLASSES)} training and {samples_per_class_val * len(CLASSES)} validation images across {len(CLASSES)} categories.")
    print(f"✅ Created presentation demo images in '{samples_path.resolve()}'.")


def download_trashnet(target_dir: str = "dataset/trashnet_raw"):
    """
    Downloads the Stanford TrashNet benchmark dataset repository if user has internet connectivity.
    Dataset source: Gary Thung and Mindy Yang (Stanford University)
    """
    url = "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    zip_dest = target_path / "trashnet.zip"
    
    print(f"📥 Attempting to download TrashNet dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_dest)
        print("📦 Extracting TrashNet archive...")
        with zipfile.ZipFile(zip_dest, "r") as zip_ref:
            zip_ref.extractall(target_path)
        print("✅ TrashNet downloaded and extracted successfully.")
    except Exception as e:
        print(f"⚠️ Could not automatically fetch external zip ({e}). Using generated/local municipal dataset.")


if __name__ == "__main__":
    setup_dataset_structure()
