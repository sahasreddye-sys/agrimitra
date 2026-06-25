"""
predict.py
AgriMitra — Single-image inference with Grad-CAM.

Used by the Streamlit app (app/main.py) and can be run standalone:
    python src/predict.py --image path/to/leaf.jpg \
                          --checkpoint models/efficientnet_b0_plantvillage_XXXXXXXX.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from train import build_model, get_transforms
from utils import (ROOT, MODELS_DIR, latest_checkpoint,
                   parse_crop_and_disease, confidence_label, is_beta_crop,
                   load_translations)


# ── Image loading ──────────────────────────────────────────────────────────

def load_image(image_source) -> Image.Image:
    """Accept a file path (str/Path) or a PIL Image directly."""
    if isinstance(image_source, (str, Path)):
        return Image.open(image_source).convert("RGB")
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image_source)}")


def preprocess(img: Image.Image, image_size: int = 224) -> torch.Tensor:
    transform = get_transforms(image_size, "val")
    return transform(img).unsqueeze(0)   # → [1, C, H, W]


# ── Prediction ─────────────────────────────────────────────────────────────

def predict(image_source, checkpoint_path=None, top_k: int = 3, lang: str = "telugu"):
    """
    Run inference on a single image.

    Returns a dict:
    {
        "crop":        "Tomato",
        "disease":     "Early Blight",
        "confidence":  "High",
        "probability": 0.94,
        "top_k": [
            {"class": "Tomato___Early_blight", "crop": "Tomato",
             "disease": "Early Blight", "prob": 0.94,
             "telugu_disease": "...", "treatment": "..."},
            ...
        ],
        "is_beta": False,
        "cam": <np.ndarray | None>,   # Grad-CAM heat map (H×W, values 0-1)
        "raw_class": "Tomato___Early_blight",
    }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    if checkpoint_path is None:
        checkpoint_path = latest_checkpoint()
    if checkpoint_path is None:
        raise FileNotFoundError("No model checkpoint found in models/. "
                                "Run: python src/train.py --dataset plantvillage")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name  = ckpt["model_name"]
    class_names = ckpt["class_names"]
    num_classes = ckpt["num_classes"]

    model = build_model(model_name, num_classes, freeze_backbone=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    # ── Image → tensor ───────────────────────────────────────────────────────
    img = load_image(image_source)
    tensor = preprocess(img).to(device)

    # ── Inference ────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]   # [num_classes]

    top_probs, top_idxs = probs.topk(min(top_k, num_classes))

    # ── Grad-CAM ─────────────────────────────────────────────────────────────
    cam_array = None
    try:
        from evaluate import GradCAM, get_gradcam_layer
        tensor_grad = preprocess(img).to(device).requires_grad_(True)
        model_eval = build_model(model_name, num_classes, freeze_backbone=False)
        model_eval.load_state_dict(ckpt["state_dict"])
        model_eval = model_eval.to(device)
        target_layer = get_gradcam_layer(model_name)
        cam_gen = GradCAM(model_eval, target_layer)
        cam_array = cam_gen.generate(tensor_grad, top_idxs[0].item())
    except Exception:
        pass   # Grad-CAM is a bonus — don't crash the app if it fails

    # ── Build result ─────────────────────────────────────────────────────────
    translations = load_translations(lang)

    top_k_results = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_idxs.cpu().numpy()):
        cls = class_names[idx]
        crop, disease = parse_crop_and_disease(cls)
        top_k_results.append({
            "class":         cls,
            "crop":          crop,
            "disease":       disease,
            "prob":          float(prob),
            f"{lang}_disease": translations.get(cls, {}).get("disease_name", disease),
            "treatment":     translations.get(cls, {}).get("treatment", ""),
            "prevention":    translations.get(cls, {}).get("prevention", ""),
        })

    best   = top_k_results[0]
    result = {
        "crop":        best["crop"],
        "disease":     best["disease"],
        "confidence":  confidence_label(best["prob"], best["crop"]),
        "probability": best["prob"],
        "top_k":       top_k_results,
        "is_beta":     is_beta_crop(best["crop"]),
        "cam":         cam_array,
        "raw_class":   best["class"],
        "model":       model_name,
        "lang":        lang,
    }
    return result


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AgriMitra single-image inference")
    parser.add_argument("--image",      required=True, help="Path to leaf image")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pt checkpoint (default: latest in models/)")
    parser.add_argument("--top_k",      type=int, default=3)
    parser.add_argument("--lang",       default="telugu",
                        choices=["telugu", "hindi", "gujarati"])
    args = parser.parse_args()

    result = predict(args.image, args.checkpoint, args.top_k, args.lang)

    print(f"\n🌿 Crop:       {result['crop']}")
    print(f"🦠 Disease:    {result['disease']}")
    print(f"📊 Confidence: {result['confidence']} ({result['probability']*100:.1f}%)")

    if result["is_beta"]:
        print("⚠  Beta crop — community-collected data, limited accuracy.")

    print(f"\nTop {len(result['top_k'])} predictions:")
    for i, r in enumerate(result["top_k"], 1):
        print(f"  {i}. {r['crop']} / {r['disease']}  ({r['prob']*100:.1f}%)")
        if r.get("treatment"):
            print(f"     💊 {r['treatment'][:120]} …")

    if result["cam"] is not None:
        print("\n✅ Grad-CAM generated (cam key in result dict).")


if __name__ == "__main__":
    main()
