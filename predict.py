"""
Inference & Explainability Module for Waste Segregation Classifier
Author: AGGIBADALA VENKATA DURGA NAGA PRASAD (23ME1A5402)
Guide: Ms. G. Hema Sri

Provides:
- Standalone waste image prediction
- Top-K probability scores
- Explainable AI (Grad-CAM Activation Heatmaps) showing visual areas influencing the CNN decision
- Smart-bin color recommendation & recyclability instructions
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.cm as cm

from model import build_model, WASTE_CLASSES, CATEGORY_METADATA


class GradCAM:
    """Computes Gradient-weighted Class Activation Map for model interpretability."""
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        score = output[0, class_idx]
        score.backward(retain_graph=True)
        
        # Pool gradients across spatial dimensions
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight activations by gradients
        activations = self.activations[0]
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val != 0:
            heatmap = heatmap / max_val
            
        return heatmap


def load_classifier(checkpoint_path: str = "waste_classifier_model.pth", device: torch.device = None):
    """Loads trained checkpoint or creates default initialized model if checkpoint does not exist."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model_type = checkpoint.get("model_type", "custom")
        class_names = checkpoint.get("class_names", WASTE_CLASSES)
        model = build_model(model_name=model_type, num_classes=len(class_names), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Fallback to pre-initialized weights
        model_type = "custom"
        class_names = WASTE_CLASSES
        model = build_model(model_name="custom", num_classes=len(class_names), pretrained=False)
        checkpoint = None

    model = model.to(device)
    model.eval()
    return model, class_names, checkpoint


def predict_waste_image(
    image: Image.Image,
    model_path: str = "waste_classifier_model.pth",
    compute_gradcam: bool = True
) -> Dict[str, Any]:
    """
    Evaluates an input image and returns classification metadata,
    probabilities, smart bin instructions, and Grad-CAM visualization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, checkpoint = load_classifier(model_path, device)
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    rgb_image = image.convert("RGB")
    tensor_input = preprocess(rgb_image).unsqueeze(0).to(device)
    
    # Enable gradient tracking for Grad-CAM
    tensor_input.requires_grad = True

    outputs = model(tensor_input)
    probabilities = F.softmax(outputs, dim=1).squeeze().cpu().detach().numpy()
    
    top_idx = int(np.argmax(probabilities))
    predicted_class = class_names[top_idx]
    confidence = float(probabilities[top_idx])
    
    all_scores = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    metadata = CATEGORY_METADATA.get(predicted_class, CATEGORY_METADATA["trash"])
    
    result = {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": all_scores,
        "metadata": metadata,
        "gradcam_overlay": None,
        "heatmap": None
    }
    
    # Compute Explainable AI Heatmap
    if compute_gradcam:
        try:
            target_layer = model.get_last_conv_layer()
            cam = GradCAM(model, target_layer)
            heatmap = cam.generate(tensor_input, top_idx)
            
            # Upsample heatmap to match original image dimensions
            heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(rgb_image.size, Image.BILINEAR)
            heatmap_np = np.array(heatmap_resized) / 255.0
            
            # Apply Jet colormap
            colormap = cm.get_cmap("jet")
            colored_heatmap = colormap(heatmap_np)[:, :, :3]  # RGB
            colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
            
            # Alpha blend with original image
            orig_np = np.array(rgb_image)
            overlay_np = (orig_np * 0.6 + colored_heatmap * 0.4).astype(np.uint8)
            
            result["gradcam_overlay"] = Image.fromarray(overlay_np)
            result["heatmap"] = Image.fromarray(colored_heatmap)
        except Exception as e:
            # Grad-CAM is optional enhancement, keep robust
            result["gradcam_error"] = str(e)
            
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Waste Item Category")
    parser.add_argument("--image", type=str, required=True, help="Path to waste item image")
    parser.add_argument("--model", type=str, default="waste_classifier_model.pth", help="Model checkpoint path")
    args = parser.parse_args()

    test_img = Image.open(args.image)
    res = predict_waste_image(test_img, model_path=args.model)
    
    print("\n--- CLASSIFICATION RESULT ---")
    print(f"Predicted Waste: {res['metadata']['label']} ({res['metadata']['icon']})")
    print(f"Confidence:     {res['confidence'] * 100:.2f}%")
    print(f"Recyclability:  {res['metadata']['category_type']}")
    print(f"Designated Bin: {res['metadata']['bin_name']}")
    print(f"Disposal Guide: {res['metadata']['disposal_instructions']}")
    print(f"Impact:         {res['metadata']['carbon_saving']}")
