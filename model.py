"""
Waste Segregation Image Classifier - CNN Architecture & Model Definition
Author: AGGIBADALA VENKATA DURGA NAGA PRASAD (23ME1A5402)
Guide: Ms. G. Hema Sri
Domain: Deep Learning & Computer Vision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Any, Tuple

# Class Categories and Domain Metadata for Smart-Bin Assistant
WASTE_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

CATEGORY_METADATA: Dict[str, Dict[str, Any]] = {
    "cardboard": {
        "label": "Cardboard",
        "category_type": "Recyclable",
        "bin_color": "#1E88E5",  # Blue
        "bin_name": "Blue Bin (Paper & Cardboard)",
        "icon": "📦",
        "decomposition_time": "2 months",
        "disposal_instructions": "Flatten boxes to save space. Keep dry and free of grease or food residues.",
        "carbon_saving": "Saving 1 ton of cardboard saves 17 mature trees and 7,000 gallons of water.",
        "tips": "Remove excess adhesive tape and plastic packaging bubbles before discarding."
    },
    "glass": {
        "label": "Glass",
        "category_type": "Recyclable",
        "bin_color": "#43A047",  # Green
        "bin_name": "Green Bin (Glass)",
        "icon": "🍾",
        "decomposition_time": "1,000,000+ years (indefinite)",
        "disposal_instructions": "Rinse remaining liquids. Remove plastic caps/lids. Handle carefully to prevent breakage.",
        "carbon_saving": "100% recyclable indefinitely without quality degradation; saves 30% energy vs new glass.",
        "tips": "Do not mix light bulbs, mirrors, or ceramic cookware with container glass."
    },
    "metal": {
        "label": "Metal",
        "category_type": "Recyclable",
        "bin_color": "#FDD835",  # Yellow
        "bin_name": "Yellow Bin (Metals & Cans)",
        "icon": "🥫",
        "decomposition_time": "50 - 500 years",
        "disposal_instructions": "Rinse beverage and food residue. Aluminum cans can be crushed to save volume.",
        "carbon_saving": "Recycling aluminum requires 95% less energy than producing aluminum from bauxite ore.",
        "tips": "Steel food cans, soda cans, aluminum foil (clean) are all accepted."
    },
    "paper": {
        "label": "Paper",
        "category_type": "Recyclable",
        "bin_color": "#1E88E5",  # Blue
        "bin_name": "Blue Bin (Paper & Cardboard)",
        "icon": "📰",
        "decomposition_time": "2 to 6 weeks",
        "disposal_instructions": "Keep clean and dry. Keep newspapers, print sheets, and notebooks segregated.",
        "carbon_saving": "1 ton of recycled paper preserves 17 trees and prevents 1.5 tons of CO2 emissions.",
        "tips": "Shredded paper, magazines, notebook pages are recyclable. Wax/plastic coated paper is not."
    },
    "plastic": {
        "label": "Plastic",
        "category_type": "Recyclable",
        "bin_color": "#FB8C00",  # Orange
        "bin_name": "Orange Bin (Plastics - PET, HDPE)",
        "icon": "🧴",
        "decomposition_time": "450 - 1000 years",
        "disposal_instructions": "Empty and rinse out fluids. Check resin identification code (PET 1, HDPE 2 preferred).",
        "carbon_saving": "Recycling 1 ton of plastic saves ~5,774 kWh of electricity and 16.3 barrels of oil.",
        "tips": "Single-use thin bags should be deposited at specialized plastic bag collection points."
    },
    "trash": {
        "label": "Trash / Organic / Non-Recyclable",
        "category_type": "Non-Recyclable / Compost",
        "bin_color": "#757575",  # Gray / Black
        "bin_name": "Gray Bin (Residual Waste & Landfill)",
        "icon": "🗑️",
        "decomposition_time": "Varies (Biodegradable to centuries)",
        "disposal_instructions": "Wrap securely if hazardous. Divert food scraps and vegetable peels to wet/compost bins.",
        "carbon_saving": "Composting organic matter eliminates methane generation in anaerobic landfills.",
        "tips": "Food waste, soiled paper napkins, broken ceramics, composites go here."
    }
}


class CustomWasteCNN(nn.Module):
    """
    Tailored Deep Convolutional Neural Network designed for Municipal Waste Classification.
    Architecture:
      - 4 Feature Extraction Blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout)
      - Adaptive Average Pooling
      - Fully Connected Dense Classifier with Dropout regularization to prevent overfitting.
    """
    def __init__(self, num_classes: int = 6, dropout_rate: float = 0.3):
        super(CustomWasteCNN, self).__init__()
        
        # Block 1: 3 -> 32 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        
        # Block 2: 32 -> 64 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate * 0.7)
        )
        
        # Block 3: 64 -> 128 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )

        # Block 4: 128 -> 256 channels (High-level semantic features)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dense classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def get_last_conv_layer(self) -> nn.Module:
        return self.block4[0]


class MobileNetV2WasteClassifier(nn.Module):
    """
    Transfer Learning Model based on MobileNetV2.
    Ideal for real-time edge devices (Smart Bins with Raspberry Pi / Jetson Nano)
    due to inverted residual blocks and depthwise separable convolutions.
    """
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super(MobileNetV2WasteClassifier, self).__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v2(weights=weights)
        
        self.features = base_model.features
        in_features = base_model.classifier[1].in_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_last_conv_layer(self) -> nn.Module:
        return self.features[-1]


def build_model(model_name: str = "custom", num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    """Factory helper to build selected model"""
    if model_name.lower() in ["mobilenet", "mobilenet_v2", "transfer"]:
        return MobileNetV2WasteClassifier(num_classes=num_classes, pretrained=pretrained)
    else:
        return CustomWasteCNN(num_classes=num_classes)
