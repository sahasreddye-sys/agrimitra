"""
data_prep.py
AgriMitra — Download, clean, split, and augment datasets.

Usage:
    python src/data_prep.py --dataset plantvillage
    python src/data_prep.py --dataset all
"""

import os
import argparse
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
RAW_DIR      = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
COLLECTED_DIR = ROOT / "data" / "collected"

TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# TEST = remainder (0.15)

IMAGE_SIZE   = (224, 224)   # EfficientNet-B0 input
SEED         = 42

# ── Dataset registry ───────────────────────────────────────────────────────
# Each entry describes how to obtain the dataset.
# "hf"     → HuggingFace datasets library
# "kaggle" → Kaggle API  (requires ~/.kaggle/kaggle.json)
# "manual" → user must place files in data/raw/<name>/ themselves

DATASETS = {
    "plantvillage": {
        "source": "hf",
        "hf_name": "mohanty/PlantVillage",
        "crops": ["Tomato", "Potato", "Pepper,_bell"],   # filter to these
        "notes": "Primary backbone — 25k+ images for tomato/potato/pepper.",
    },
    "chilli": {
        "source": "manual",
        "raw_subdir": "chilli_mendeley",
        "notes": "8,814 images. Download from: "
                 "https://data.mendeley.com/datasets/tm3v4zmh7c/1",
    },
    "bangladesh_12k": {
        "source": "manual",
        "raw_subdir": "bangladesh_12k",
        "notes": "Bitter gourd, bottle gourd, eggplant, cucumber. "
                 "Download from: https://data.mendeley.com/datasets/v46jkbbzv3/2",
    },
    "okra": {
        "source": "manual",
        "raw_subdir": "okra_diseasenet",
        "notes": "Indian field images. "
                 "Download from: https://data.mendeley.com/datasets/nh7zk4hv8z/1",
    },
    "mango": {
        "source": "manual",
        "raw_subdir": "mangoleafbd",
        "notes": "6,000+ images. "
                 "Download from: https://data.mendeley.com/datasets/hxsnvwty3r/1",
    },
    "malabar_spinach": {
        "source": "manual",
        "raw_subdir": "malabar_spinach",
        "notes": "20,173 images. "
                 "Download from: https://data.mendeley.com/datasets/sy69db2nz5/1",
    },
    "ash_gourd": {
        "source": "manual",
        "raw_subdir": "ash_gourd",
        "notes": "2,676 images. "
                 "Download from: https://data.mendeley.com/datasets/zj4th6xvdp/2",
    },
    "jackfruit": {
        "source": "kaggle",
        "kaggle_dataset": "shuvokumarbasak4004/jackfruit-leaf-diseases",
        "notes": "38k images. Requires Kaggle API key.",
    },
    "coconut": {
        "source": "manual",
        "raw_subdir": "coconut",
        "notes": "5,798 images. "
                 "Download from: https://data.mendeley.com/datasets/gh56wbsnj5/1",
    },
    "papaya": {
        "source": "manual",
        "raw_subdir": "papaya",
        "notes": "3,626 raw images. "
                 "Download from: https://data.mendeley.com/datasets/3kwgxg4stb/1",
    },
    # Community-collected crops (no public data)
    "ivy_gourd":   {"source": "collected", "notes": "Dondakaya — community photos only."},
    "gongura":     {"source": "collected", "notes": "Roselle leaves — community photos only."},
    "curry_leaves":{"source": "collected", "notes": "Community photos only."},
    "thotakura":   {"source": "collected", "notes": "Amaranth — community photos only."},
}


# ── Download helpers ───────────────────────────────────────────────────────

def download_plantvillage(out_dir: Path):
    """Stream PlantVillage from HuggingFace and save as class folders."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Run: pip install datasets")

    print("⬇  Downloading PlantVillage from HuggingFace …")
    ds = load_dataset("mohanty/PlantVillage", split="train")
    target_crops = DATASETS["plantvillage"]["crops"]

    saved = 0
    for item in ds:
        label: str = item["label"]   # e.g. "Tomato___Early_blight"
        # Keep only target crop classes
        if not any(label.startswith(c) for c in target_crops):
            continue
        class_dir = out_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        img_path = class_dir / f"{saved:06d}.jpg"
        item["image"].save(img_path)
        saved += 1
        if saved % 500 == 0:
            print(f"   … {saved} images saved")

    print(f"✅ PlantVillage: {saved} images → {out_dir}")


def download_kaggle(dataset_slug: str, out_dir: Path):
    """Download a Kaggle dataset. Requires ~/.kaggle/kaggle.json."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise SystemExit("Run: pip install kaggle  and place kaggle.json in ~/.kaggle/")

    out_dir.mkdir(parents=True, exist_ok=True)
    os.system(f"kaggle datasets download -d {dataset_slug} -p {out_dir} --unzip")
    print(f"✅ Kaggle '{dataset_slug}' → {out_dir}")


# ── Image utilities ────────────────────────────────────────────────────────

def is_valid_image(path: Path) -> bool:
    """Return True if the file can be opened as an RGB image."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def clean_and_resize(src_dir: Path, dst_dir: Path):
    """
    Walk src_dir, skip corrupt files, resize to IMAGE_SIZE, save to dst_dir
    preserving the class-folder structure.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    total, skipped = 0, 0

    for img_path in src_dir.rglob("*"):
        if img_path.suffix.lower() not in exts:
            continue
        if not is_valid_image(img_path):
            skipped += 1
            continue

        rel = img_path.relative_to(src_dir)
        out_path = dst_dir / rel.parent / (rel.stem + ".jpg")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(img_path) as img:
            img = img.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
            img.save(out_path, "JPEG", quality=95)
        total += 1

    print(f"   Cleaned: {total} images kept, {skipped} skipped (corrupt/unreadable)")
    return total


# ── Train / Val / Test split ───────────────────────────────────────────────

def split_dataset(src_dir: Path, split_dir: Path):
    """
    Split class folders in src_dir into train/val/test subfolders inside split_dir.
    Preserves class balance.
    """
    random.seed(SEED)
    splits = {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": 1 - TRAIN_RATIO - VAL_RATIO}

    for class_dir in sorted(src_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        images = list(class_dir.glob("*.jpg"))
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        buckets  = {
            "train": images[:n_train],
            "val":   images[n_train : n_train + n_val],
            "test":  images[n_train + n_val :],
        }

        for split_name, files in buckets.items():
            dest = split_dir / split_name / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest / f.name)

        print(f"   {class_dir.name}: {n} images → "
              f"train {len(buckets['train'])} / "
              f"val {len(buckets['val'])} / "
              f"test {len(buckets['test'])}")


# ── Augmentation (offline — saved to disk) ────────────────────────────────

def augment_training_set(train_dir: Path, target_per_class: int = 1000):
    """
    For classes with fewer than target_per_class images, apply random
    augmentations until we reach the target. Saves augmented images in-place.

    Augmentations: horizontal flip, vertical flip, rotation (±30°),
                   brightness/contrast jitter, random crop + resize.
    """
    from PIL import ImageEnhance, ImageOps

    def augment_one(img: Image.Image) -> Image.Image:
        ops = [
            lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
            lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
            lambda x: x.rotate(random.uniform(-30, 30), expand=False,
                               fillcolor=(0, 0, 0)),
            lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.7, 1.3)),
            lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.8, 1.2)),
            lambda x: ImageOps.autocontrast(x),
        ]
        chosen = random.sample(ops, k=random.randint(1, 3))
        for op in chosen:
            img = op(img)
        return img

    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        existing = list(class_dir.glob("*.jpg"))
        n = len(existing)
        if n >= target_per_class:
            continue

        needed = target_per_class - n
        print(f"   Augmenting {class_dir.name}: {n} → {target_per_class} (+{needed})")
        aug_idx = 0
        while aug_idx < needed:
            src = random.choice(existing)
            with Image.open(src) as img:
                aug = augment_one(img.convert("RGB"))
                out_name = f"aug_{aug_idx:05d}.jpg"
                aug.save(class_dir / out_name, "JPEG", quality=90)
            aug_idx += 1


# ── Main ──────────────────────────────────────────────────────────────────

def process_dataset(name: str):
    info = DATASETS.get(name)
    if info is None:
        raise ValueError(f"Unknown dataset: '{name}'. "
                         f"Choose from: {list(DATASETS.keys())}")

    source = info["source"]
    raw_dir = RAW_DIR / name
    clean_dir = PROCESSED_DIR / name / "cleaned"
    split_dir = PROCESSED_DIR / name / "split"

    print(f"\n{'─'*60}")
    print(f"Dataset: {name}  |  source: {source}")
    if "notes" in info:
        print(f"Notes:   {info['notes']}")
    print(f"{'─'*60}")

    # 1. Download
    if source == "hf":
        download_plantvillage(raw_dir)
    elif source == "kaggle":
        download_kaggle(info["kaggle_dataset"], raw_dir)
    elif source == "manual":
        subdir = raw_dir / info.get("raw_subdir", name)
        if not subdir.exists():
            print(f"⚠  Manual download required.\n"
                  f"   Place files in: {subdir}\n"
                  f"   Then re-run this script.")
            return
    elif source == "collected":
        src = COLLECTED_DIR / name
        if not src.exists() or not any(src.rglob("*.jpg")):
            print(f"⚠  No community photos yet for '{name}'.\n"
                  f"   Place photos in: {src}")
            return
        raw_dir = src   # use collected folder directly

    # 2. Clean + resize
    print(f"\n📐 Cleaning & resizing → {clean_dir}")
    clean_dir.mkdir(parents=True, exist_ok=True)
    n = clean_and_resize(raw_dir, clean_dir)
    if n == 0:
        print("❌ No images found after cleaning. Check your raw folder.")
        return

    # 3. Split
    print(f"\n✂  Splitting (70/15/15) → {split_dir}")
    split_dataset(clean_dir, split_dir)

    # 4. Augment training set to at least 1000 images per class
    print(f"\n🔀 Augmenting training set …")
    augment_training_set(split_dir / "train")

    print(f"\n✅ '{name}' ready at: {split_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="AgriMitra data preparation pipeline")
    parser.add_argument(
        "--dataset",
        default="plantvillage",
        help=f"Dataset name or 'all'. Choices: {list(DATASETS.keys())}",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        for name in DATASETS:
            try:
                process_dataset(name)
            except Exception as e:
                print(f"⚠  Skipping '{name}': {e}")
    else:
        process_dataset(args.dataset)


if __name__ == "__main__":
    main()
