"""
Evaluation Script for Demonstration Benchmark Samples
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

from pathlib import Path
from PIL import Image
from predict import predict_waste_image

def main():
    samples_dir = Path("samples")
    sample_files = sorted(list(samples_dir.glob("demo_*.jpg")))

    print("=" * 80)
    print("🧪 SMART-BIN WASTE SEGREGATION BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"{'Input Sample Image':<22} | {'Predicted Category':<18} | {'Confidence':<12} | {'Designated Bin'}")
    print("-" * 80)

    correct = 0
    for img_path in sample_files:
        true_label = img_path.stem.replace("demo_", "")
        img = Image.open(img_path)
        res = predict_waste_image(img, compute_gradcam=False)
        pred_label = res["predicted_class"]
        conf = res["confidence"] * 100
        bin_name = res["metadata"]["bin_name"]

        is_match = (true_label == pred_label)
        if is_match:
            correct += 1
        status_icon = "✅" if is_match else "❌"

        print(f"{img_path.name:<22} | {pred_label.capitalize():<18} | {conf:>6.2f}% {status_icon} | {bin_name}")

    print("-" * 80)
    print(f"🎯 Total Benchmark Accuracy: {correct}/{len(sample_files)} ({correct/len(sample_files)*100:.1f}%)")
    print("=" * 80)

if __name__ == "__main__":
    main()
