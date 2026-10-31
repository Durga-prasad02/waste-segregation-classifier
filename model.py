"""
Waste Segregation Classifier - Neural Network Architecture
Author: AGGIBADALA VENKATA DURGA NAGA PRASAD (23ME1A5402)
Guide: Ms. G. Hema Sri
Domain: Deep Learning & Computer Vision
"""

import torch
import torch.nn as nn
from typing import Dict, Any

WASTE_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

CATEGORY_METADATA: Dict[str, Dict[str, Any]] = {
    "cardboard": {
        "label": "Cardboard",
        "category_type": "Recyclable",
        "bin_color": "#1976d2",  # Blue
        "bin_name": "Blue Bin (Paper & Cardboard)",
        "icon": "📦",
        "decomposition_time": "2 months",
        "carbon_saving": "Saves 24% energy vs. virgin cardboard production",
        "disposal_instructions": "Flatten boxes completely before putting into bin. Keep dry and free of greasy food contamination.",
        "tips": "Remove excess plastic adhesive tape when possible to assist pulping mills."
    },
    "glass": {
        "label": "Glass",
        "category_type": "Recyclable",
        "bin_color": "#2e7d32",  # Green
        "bin_name": "Green Bin (Glass Containers)",
        "icon": "🍾",
        "decomposition_time": "1,000,000+ years (indefinite)",
        "carbon_saving": "Recycling 1 ton of glass saves 315 kg of CO2",
        "disposal_instructions": "Rinse out residual liquids and solids. Remove caps and metal rings.",
        "tips": "Glass is 100% recyclable infinitely without loss of chemical purity or quality."
    },
    "metal": {
        "label": "Metal / Cans",
        "category_type": "Recyclable",
        "bin_color": "#fbc02d",  # Yellow
        "bin_name": "Yellow Bin (Metals & Cans)",
        "icon": "🥫",
        "decomposition_time": "50 - 200 years (Aluminum / Tin)",
        "carbon_saving": "Recycling aluminum requires 95% less energy than mining bauxite",
        "disposal_instructions": "Rinse clean and lightly crush cans to conserve smart-bin volume.",
        "tips": "Recycling a single aluminum beverage can saves enough energy to run a TV for 3 hours."
    },
    "paper": {
        "label": "Paper",
        "category_type": "Recyclable",
        "bin_color": "#0288d1",  # Light Blue
        "bin_name": "Blue Bin (Paper & Cardboard)",
        "icon": "📄",
        "decomposition_time": "2 - 6 weeks",
        "carbon_saving": "Recycling 1 ton of paper saves 17 trees and 26,500 liters of water",
        "disposal_instructions": "Keep paper clean and dry. Avoid shredded paper in standard bins without bundling.",
        "tips": "Ensure no greasy food contamination (such as oily pizza boxes)."
    },
    "plastic": {
        "label": "Plastic",
        "category_type": "Recyclable",
        "bin_color": "#ef6c00",  # Orange
        "bin_name": "Orange Bin (Plastics - PET, HDPE)",
        "icon": "🧴",
        "decomposition_time": "450 - 500 years",
        "carbon_saving": "Recycling plastics saves 5.774 kWh of energy per kilogram",
        "disposal_instructions": "Empty all liquids completely. Replace the screw cap so it gets processed together.",
        "tips": "Check plastic resin code (RIC 1 PET and RIC 2 HDPE are highest value recyclables)."
    },
    "trash": {
        "label": "Trash / Organic",
        "category_type": "Non-Recyclable",
        "bin_color": "#616161",  # Dark Gray
        "bin_name": "Gray Bin (Residual Waste & Landfill)",
        "icon": "🗑️",
        "decomposition_time": "Variable (Months for organic, indefinite for composite wrappers)",
        "carbon_saving": "Composting organic fraction eliminates methane generation in anaerobic landfills",
        "disposal_instructions": "Wrap securely if hazardous. Divert food scraps and vegetable peels to wet/compost bins.",
        "tips": "Separating wet organic waste prevents cross-contamination of high-value recyclables."
    }
}


class ConvBlock(nn.Module):
    """Convolutional Block with BatchNorm, ReLU, MaxPool, and Spatial Dropout."""
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.25):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class WasteClassifierCNN(nn.Module):
    """
    Tailored Deep Convolutional Neural Network for Municipal Waste Image Classification.
    Accepts 224x224 RGB input images and outputs logits for the 6 waste categories.
    """
    def __init__(self, num_classes: int = 6):
        super().__init__()
        # 4 Sequential Convolutional Feature Extraction Blocks
        self.block1 = ConvBlock(in_channels=3, out_channels=32, dropout_p=0.15)
        self.block2 = ConvBlock(in_channels=32, out_channels=64, dropout_p=0.20)
        self.block3 = ConvBlock(in_channels=64, out_channels=128, dropout_p=0.25)
        self.block4 = ConvBlock(in_channels=128, out_channels=256, dropout_p=0.30)
        
        # Adaptive Global Average Pooling converts spatial (B, 256, H, W) -> (B, 256, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


def build_model(model_name: str = "custom", num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    """Builds and returns the neural network model."""
    if model_name.lower() == "custom":
        return WasteClassifierCNN(num_classes=num_classes)
    elif model_name.lower() == "mobilenet_v2":
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
