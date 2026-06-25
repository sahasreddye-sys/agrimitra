"""
evaluate.py
AgriMitra — Evaluation: per-class metrics, confusion matrix, Grad-CAM.

Usage:
    python src/evaluate.py --checkpoint models/efficientnet_b0_plantvillage_XXXXXXXX.pt
                           --dataset plantvillage
                           --gradcam                   # generate Grad-CAM images
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)

from train import build_model, get_transforms, SUPPORTED_MODELS
from utils import ROOT, MODELS_DIR, parse_crop_and_disease

EVAL_DIR = ROOT / "models" / "eval"


# ── Grad-CAM (EfficientNet / any CNN) ─────────────────────────────────────

class GradCAM:
    """
    Lightweight Grad-CAM that doesn't require the grad-cam package.
    Works on any CNN with a named target layer.
    """
    def __init__(self, model, target_layer_name: str):
        self.model = model
        self.gradients = None
        self.activations = None

        # Find the target layer by name
        target = dict(model.named_modules()).get(target_layer_name)
        if target is None:
            raise ValueError(f"Layer '{target_layer_name}' not found in model. "
                             f"Available: {list(dict(model.named_modules()).keys())[:20]}")

        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()

        pooled_grads = self.gradients.mean(dim=[0, 2, 3])  # GAP over spatial dims
        cam = (pooled_grads[:, None, None] * self.activations[0]).sum(0)
        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy()


def get_gradcam_layer(model_name: str) -> str:
    """Return the last convolutional layer name for each supported architecture."""
    mapping = {
        "efficientnet_b0":    "features.8.0",
        "efficientnet_b3":    "features.8.0",
        "densenet121":        "features.norm5",
        "mobilenet_v3_small": "features.12.0",
    }
    return mapping.get(model_name, "features")


def save_gradcam_overlay(img_tensor, cam: np.ndarray, pred_label: str,
                          true_label: str, save_path: Path):
    """Overlay the Grad-CAM heatmap on the original image and save."""
    from PIL import Image
    import cv2

    # De-normalise image
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * std + mean).clip(0, 1)

    # Resize CAM to image size
    h, w = img_np.shape[:2]
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0

    heatmap = plt.cm.jet(cam_resized)[:, :, :3]
    overlay = 0.5 * img_np + 0.5 * heatmap

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(axes,
                                [img_np, cam_resized, overlay],
                                ["Original", "Grad-CAM", "Overlay"]):
        ax.imshow(data, cmap="jet" if title == "Grad-CAM" else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    status = "✓ Correct" if pred_label == true_label else f"✗ Predicted: {pred_label}"
    fig.suptitle(f"True: {true_label}  |  {status}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_model(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = probs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return (np.array(all_preds), np.array(all_labels), np.array(all_probs))


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    # Normalise to percentage
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n = len(class_names)
    fig_size = max(10, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    short_names = [parse_crop_and_disease(c)[1] for c in class_names]

    sns.heatmap(cm_norm, annot=(n <= 20), fmt=".0%", cmap="Blues",
                xticklabels=short_names, yticklabels=short_names,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Confusion Matrix (normalised)", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion matrix → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AgriMitra evaluation")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--dataset",    required=True,
                        help="Dataset name (must match processed/ subdirectory)")
    parser.add_argument("--split",      default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--gradcam",    action="store_true",
                        help="Generate Grad-CAM overlays for misclassified images")
    parser.add_argument("--gradcam_n",  type=int, default=10,
                        help="Number of Grad-CAM images to generate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model_name  = ckpt["model_name"]
    class_names = ckpt["class_names"]
    num_classes = ckpt["num_classes"]

    model = build_model(model_name, num_classes, freeze_backbone=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    print(f"✅ Loaded '{model_name}' checkpoint (val_acc: {ckpt['val_acc']:.2f}%)")

    # Load data
    split_dir = ROOT / "data" / "processed" / args.dataset / "split" / args.split
    if not split_dir.exists():
        raise SystemExit(f"No data at {split_dir}")

    ds = datasets.ImageFolder(split_dir,
                               transform=get_transforms(224, args.split))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    # Evaluate
    print(f"\n🔍 Evaluating on {args.split} set ({len(ds)} images) …")
    preds, labels, probs = evaluate_model(model, loader, device, class_names)

    acc = accuracy_score(labels, preds)
    print(f"\nOverall accuracy: {acc*100:.2f}%")
    print("\nPer-class report:")
    print(classification_report(labels, preds, target_names=class_names, digits=3))

    # Save outputs
    run_tag = ckpt_path.stem
    eval_dir = EVAL_DIR / run_tag
    eval_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(labels, preds, class_names,
                          eval_dir / "confusion_matrix.png")

    # Grad-CAM for misclassified images
    if args.gradcam:
        print(f"\n🎯 Generating Grad-CAM for up to {args.gradcam_n} "
              f"misclassified images …")
        target_layer = get_gradcam_layer(model_name)
        cam_gen = GradCAM(model, target_layer)
        gradcam_dir = eval_dir / "gradcam"

        misclassified = np.where(preds != labels)[0]
        chosen = misclassified[:args.gradcam_n]

        for idx in chosen:
            img_tensor, true_label_idx = ds[idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            img_tensor.requires_grad_(False)

            pred_idx = preds[idx]
            true_label = class_names[true_label_idx]
            pred_label = class_names[pred_idx]

            img_tensor_grad = img_tensor.clone().requires_grad_(True)
            cam = cam_gen.generate(img_tensor_grad, pred_idx)

            save_path = gradcam_dir / f"{idx:04d}_{true_label[:30]}.png"
            save_gradcam_overlay(img_tensor, cam, pred_label, true_label, save_path)

        print(f"   Grad-CAM images → {gradcam_dir}")

    print(f"\n✅ Evaluation complete. Results → {eval_dir}")


if __name__ == "__main__":
    main()
