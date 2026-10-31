"""
Inference & Explainability Module for Waste Segregation Classifier
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
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        target = output[0, class_idx]
        target.backward(retain_graph=True)
        
        # Global average pool the gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize between 0 and 1
        cam = cam.squeeze().cpu().detach().numpy()
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max > 0:
            cam = cam / cam_max
        return cam


def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def overlay_heatmap_on_image(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Tuple[Image.Image, Image.Image]:
    """Generates colored Grad-CAM overlay and raw heatmap."""
    orig_w, orig_h = image.size
    cam_resized = Image.fromarray(np.uint8(255 * cam)).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    cam_np = np.array(cam_resized) / 255.0
    
    # Jet colormap
    colormap = cm.get_cmap("jet")
    colored_cam = colormap(cam_np)[:, :, :3]
    colored_cam = np.uint8(colored_cam * 255)
    
    orig_np = np.array(image.convert("RGB"))
    blended = np.uint8(alpha * colored_cam + (1 - alpha) * orig_np)
    
    return Image.fromarray(blended), Image.fromarray(colored_cam)


def load_waste_model(model_path: str = "waste_classifier_model.pth", device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_model(model_name="custom", num_classes=len(WASTE_CLASSES), pretrained=False)
    
    path_obj = Path(model_path)
    if path_obj.exists():
        checkpoint = torch.load(str(path_obj), map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"⚠️ Model checkpoint '{model_path}' not found! Instantiating model with initialized weights.")
        
    model.to(device)
    model.eval()
    return model, device


def predict_waste_image(
    image: Image.Image,
    model_path: str = "waste_classifier_model.pth",
    compute_gradcam: bool = True
) -> Dict[str, Any]:
    """
    Analyzes an input waste image and returns classification metadata and Grad-CAM saliency.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, device = load_waste_model(model_path=model_path, device=device)
    
    transform = get_inference_transform()
    rgb_image = image.convert("RGB")
    tensor = transform(rgb_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        
    pred_idx = int(np.argmax(probabilities))
    pred_class = WASTE_CLASSES[pred_idx]
    confidence = float(probabilities[pred_idx])
    
    class_probs = {WASTE_CLASSES[i]: float(probabilities[i]) for i in range(len(WASTE_CLASSES))}
    
    gradcam_overlay = None
    heatmap = None
    if compute_gradcam:
        try:
            target_layer = model.block4.conv
            gradcam_generator = GradCAM(model, target_layer)
            tensor_for_grad = tensor.clone().detach().requires_grad_(True)
            cam = gradcam_generator.generate(tensor_for_grad, pred_idx)
            gradcam_overlay, heatmap = overlay_heatmap_on_image(rgb_image, cam)
        except Exception as e:
            print(f"⚠️ Grad-CAM could not be generated: {e}")
            
    metadata = CATEGORY_METADATA.get(pred_class, CATEGORY_METADATA["trash"])
    
    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": class_probs,
        "metadata": metadata,
        "gradcam_overlay": gradcam_overlay,
        "heatmap": heatmap
    }


def main():
    parser = argparse.ArgumentParser(description="Waste Image Classifier CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input waste image")
    parser.add_argument("--model", type=str, default="waste_classifier_model.pth", help="Path to checkpoint")
    args = parser.parse_args()
    
    img = Image.open(args.image)
    result = predict_waste_image(img, model_path=args.model, compute_gradcam=False)
    
    meta = result["metadata"]
    print("\n--- CLASSIFICATION RESULT ---")
    print(f"Predicted Waste: {meta['label']} ({meta['icon']})")
    print(f"Confidence:     {result['confidence'] * 100:.2f}%")
    print(f"Recyclability:  {meta['category_type']}")
    print(f"Designated Bin: {meta['bin_name']}")
    print(f"Disposal Guide: {meta['disposal_instructions']}")
    print(f"Impact:         {meta['carbon_saving']}")


if __name__ == "__main__":
    main()
