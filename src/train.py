"""
train.py
AgriMitra — EfficientNet-B0 transfer-learning training loop.

Usage:
    python src/train.py --dataset plantvillage
    python src/train.py --dataset plantvillage --model efficientnet_b0 --epochs 20
    python src/train.py --dataset plantvillage --model efficientnet_b3 --epochs 30

Runs on CPU, but designed for Google Colab with a T4 GPU.
Mount your Drive first:
    from google.colab import drive; drive.mount('/content/drive')
    !cd /content/drive/MyDrive/agrimitra && python src/train.py --dataset plantvillage
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import MODELS_DIR, ROOT, print_dataset_summary

# ── Config defaults ────────────────────────────────────────────────────────

DEFAULTS = {
    "model":        "efficientnet_b0",
    "epochs":       20,
    "batch_size":   32,
    "lr":           1e-3,
    "unfreeze_epoch": 5,   # unfreeze backbone after this many epochs (fine-tune)
    "image_size":   224,
    "num_workers":  2,
    "seed":         42,
}

SUPPORTED_MODELS = {
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b3": models.efficientnet_b3,
    "densenet121":     models.densenet121,
    "mobilenet_v3_small": models.mobilenet_v3_small,
}

# ── Data transforms ────────────────────────────────────────────────────────

def get_transforms(image_size: int, split: str):
    mean = [0.485, 0.456, 0.406]   # ImageNet stats
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ── Model builder ──────────────────────────────────────────────────────────

def build_model(model_name: str, num_classes: int, freeze_backbone: bool = True):
    """
    Load a pretrained model, replace its classifier head with num_classes outputs.
    If freeze_backbone=True, only the new head trains initially.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. "
                         f"Choose from {list(SUPPORTED_MODELS.keys())}")

    model_fn = SUPPORTED_MODELS[model_name]
    model = model_fn(weights="IMAGENET1K_V1")

    # Freeze all layers first
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace head for EfficientNet / MobileNet
    if "efficientnet" in model_name or "mobilenet" in model_name:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif model_name == "densenet121":
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model


def unfreeze_backbone(model):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    print("🔓 Backbone unfrozen — full fine-tuning enabled.")


# ── Training loop ──────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch} [{batch_idx*len(images)}/{len(loader.dataset)}] "
                  f"loss: {loss.item():.4f}")

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AgriMitra training")
    parser.add_argument("--dataset",   default="plantvillage")
    parser.add_argument("--model",     default=DEFAULTS["model"],
                        choices=list(SUPPORTED_MODELS.keys()))
    parser.add_argument("--epochs",    type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size",type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr",        type=float, default=DEFAULTS["lr"])
    parser.add_argument("--unfreeze_epoch", type=int,
                        default=DEFAULTS["unfreeze_epoch"])
    parser.add_argument("--image_size",type=int, default=DEFAULTS["image_size"])
    args = parser.parse_args()

    torch.manual_seed(DEFAULTS["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    split_dir = ROOT / "data" / "processed" / args.dataset / "split"
    if not split_dir.exists():
        raise SystemExit(f"No processed data at {split_dir}.\n"
                         f"Run: python src/data_prep.py --dataset {args.dataset}")

    print(f"\n📂 Dataset: {args.dataset}")
    print_dataset_summary(split_dir)

    train_ds = datasets.ImageFolder(split_dir / "train",
                                    transform=get_transforms(args.image_size, "train"))
    val_ds   = datasets.ImageFolder(split_dir / "val",
                                    transform=get_transforms(args.image_size, "val"))
    test_ds  = datasets.ImageFolder(split_dir / "test",
                                    transform=get_transforms(args.image_size, "test"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=DEFAULTS["num_workers"],
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=DEFAULTS["num_workers"])
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=DEFAULTS["num_workers"])

    num_classes = len(train_ds.classes)
    class_names = train_ds.classes
    print(f"\n🌿 Classes ({num_classes}): {class_names[:5]} …")

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_model(args.model, num_classes, freeze_backbone=True)
    model = model.to(device)
    print(f"\n🧠 Model: {args.model} | "
          f"Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ────────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_path = MODELS_DIR / f"{args.model}_{args.dataset}_{timestamp}.pt"
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"\n🏋  Starting training for {args.epochs} epochs …")
    for epoch in range(1, args.epochs + 1):

        # Unfreeze backbone after warmup
        if epoch == args.unfreeze_epoch:
            unfreeze_backbone(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch {epoch:02d}/{args.epochs} — "
              f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.2f}%  |  "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":       epoch,
                "model_name":  args.model,
                "num_classes": num_classes,
                "class_names": class_names,
                "state_dict":  model.state_dict(),
                "val_acc":     val_acc,
            }, ckpt_path)
            print(f"   💾 Saved best model (val_acc: {val_acc:.2f}%) → {ckpt_path.name}")

    # ── Test evaluation ──────────────────────────────────────────────────────
    print(f"\n📊 Loading best checkpoint for test evaluation …")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"✅ Test accuracy: {test_acc:.2f}%  |  Test loss: {test_loss:.4f}")

    # Save history
    history_path = MODELS_DIR / f"{args.model}_{args.dataset}_{timestamp}_history.json"
    with open(history_path, "w") as f:
        json.dump({**history, "test_acc": test_acc, "class_names": class_names}, f, indent=2)
    print(f"📝 Training history → {history_path.name}")


if __name__ == "__main__":
    main()
