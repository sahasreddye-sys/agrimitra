"""
utils.py
AgriMitra — shared helper functions.
"""

import json
import os
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

# ── Label / translation helpers ────────────────────────────────────────────

TRANSLATIONS_DIR = ROOT / "app" / "translations"

_translation_cache: dict[str, dict] = {}

def load_translations(lang: str) -> dict:
    """Load and cache a language JSON file. lang: 'telugu', 'hindi', 'gujarati'."""
    if lang not in _translation_cache:
        path = TRANSLATIONS_DIR / f"{lang}.json"
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            _translation_cache[lang] = json.load(f)
    return _translation_cache[lang]


def translate(key: str, lang: str = "telugu") -> str:
    """
    Return the translated string for key in the given language.
    Falls back to the key itself if not found.
    """
    data = load_translations(lang)
    return data.get(key, key)


# ── Class-name normalisation ───────────────────────────────────────────────

def normalise_class_name(raw: str) -> str:
    """
    Convert dataset folder names like 'Tomato___Early_blight' into
    a clean display string: 'Tomato — Early Blight'.
    """
    raw = raw.replace("___", " — ").replace("_", " ")
    return raw.title()


def parse_crop_and_disease(class_name: str) -> tuple[str, str]:
    """
    Split a PlantVillage-style class name into (crop, disease).
    E.g. 'Tomato___Early_blight' → ('Tomato', 'Early Blight')
         'Tomato___healthy'       → ('Tomato', 'Healthy')
    """
    if "___" in class_name:
        crop, disease = class_name.split("___", 1)
        return crop.replace("_", " ").title(), disease.replace("_", " ").title()
    return class_name.replace("_", " ").title(), "Unknown"


# ── Dataset statistics ─────────────────────────────────────────────────────

def count_images_per_class(directory: Path) -> dict[str, int]:
    """
    Walk a directory of class subfolders and return {class_name: image_count}.
    """
    counts = {}
    if not directory.exists():
        return counts
    for class_dir in sorted(directory.iterdir()):
        if not class_dir.is_dir():
            continue
        n = sum(1 for f in class_dir.rglob("*")
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        counts[class_dir.name] = n
    return counts


def print_dataset_summary(processed_dir: Path):
    """Print a quick summary of train/val/test class counts."""
    for split in ["train", "val", "test"]:
        split_dir = processed_dir / split
        if not split_dir.exists():
            continue
        counts = count_images_per_class(split_dir)
        total = sum(counts.values())
        print(f"\n{split.upper()} ({total} images, {len(counts)} classes):")
        for cls, n in counts.items():
            crop, disease = parse_crop_and_disease(cls)
            print(f"  {crop} / {disease}: {n}")


# ── Model helpers ──────────────────────────────────────────────────────────

MODELS_DIR = ROOT / "models"

def latest_checkpoint(prefix: str = "efficientnet_b0") -> Optional[Path]:
    """Return the most recently modified .h5 / .pt model file matching prefix."""
    MODELS_DIR.mkdir(exist_ok=True)
    candidates = list(MODELS_DIR.glob(f"{prefix}*.h5")) + \
                 list(MODELS_DIR.glob(f"{prefix}*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ── Confidence helpers ─────────────────────────────────────────────────────

BETA_CROPS = {
    "ivy_gourd", "gongura", "curry_leaves", "thotakura",
    "coriander", "fenugreek", "mint",
}

def is_beta_crop(crop_name: str) -> bool:
    """Return True if this crop was trained on community-collected data only."""
    return crop_name.lower().replace(" ", "_") in BETA_CROPS


def confidence_label(prob: float, crop_name: str = "") -> str:
    """
    Map a softmax probability to a human-readable confidence string.
    Beta crops get a lower effective ceiling.
    """
    if is_beta_crop(crop_name):
        if prob > 0.85:   return "Moderate (Community Beta)"
        if prob > 0.60:   return "Low (Community Beta)"
        return "Very Low (Community Beta)"
    if prob > 0.90:   return "High"
    if prob > 0.70:   return "Moderate"
    if prob > 0.50:   return "Low"
    return "Very Low — please retake photo in good light"
