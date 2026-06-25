# AgriMitra 🌿

**Multilingual plant disease detection for Indian home gardeners**

Built for the Telugu, Hindi, and Gujarati-speaking Indian diaspora community in Forsyth County and Alpharetta, Georgia.

> Congressional App Challenge GA-06 · October 2026

---

## What it does

Upload a photo of a diseased plant leaf → AgriMitra identifies the disease and shows:
- **Disease name** in Telugu, Hindi, or Gujarati
- **Grad-CAM heatmap** showing which part of the leaf the model analyzed
- **Treatment card** with organic-first recommendations (neem oil, copper fungicide)
- **Prevention tips** to stop recurrence

Crops covered include tomato, potato, green chili, okra, bitter gourd, bottle gourd, eggplant, mango, banana, papaya, coconut, and more.

Four Telugu-community crops — **Dondakaya (Ivy Gourd)**, **Gongura**, **Curry Leaves**, and **Thotakura (Amaranth)** — have no public disease datasets anywhere in the world. AgriMitra includes the first community-photographed labeled disease images for these crops, collected from Forsyth County home gardens and the Sri Hanuman Mandir community.

---

## Project structure

```
agrimitra/
├── data/
│   ├── raw/          # Downloaded datasets — never modify
│   ├── processed/    # Cleaned, split, augmented datasets
│   └── collected/    # Community-photographed images
├── src/
│   ├── data_prep.py  # Download → clean → split → augment
│   ├── train.py      # EfficientNet-B0 training loop
│   ├── evaluate.py   # Metrics, confusion matrix, Grad-CAM
│   ├── predict.py    # Single-image inference
│   └── utils.py      # Shared helpers
├── app/
│   ├── main.py       # Streamlit app
│   └── translations/ # Telugu, Hindi, Gujarati JSON files
├── models/           # Saved checkpoints (not in git)
├── tests/
└── requirements.txt
```

---

## Quickstart (Google Colab)

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Navigate to project
%cd /content/drive/MyDrive/agrimitra

# 3. Install dependencies
!pip install -r requirements.txt

# 4. Download PlantVillage and prep data
!python src/data_prep.py --dataset plantvillage

# 5. Train
!python src/train.py --dataset plantvillage --epochs 20
```

---

## Model

| Version | Architecture | Target |
|---------|-------------|--------|
| V1 | EfficientNet-B0 (ImageNet pretrained) | CAC submission Oct 2026 |
| V2 | EfficientNet-B3 | Expanded crops, offline TFLite |
| V3 | MobileNetV3-Small | Mobile PWA |

---

## Languages

| Language | Priority | Community |
|----------|----------|-----------|
| Telugu | 1 | ~26% of GA Indian population; fastest-growing subgroup |
| Hindi | 2 | ~28%; pan-Indian bridge language |
| Gujarati | 3 | ~18%; strong home-garden culture |

---

## Original dataset contribution

| Crop | Telugu Name | Status |
|------|-------------|--------|
| Ivy Gourd | దొండకాయ (Dondakaya) | First public disease dataset |
| Gongura | గోంగూర | First public disease dataset |
| Curry Leaves | కరివేపాకు | First public disease dataset |
| Amaranth | తోటకూర (Thotakura) | First public disease dataset |

Community images collected at Sri Hanuman Mandir, Forsyth County Indian community gardens, and GATA events. Dataset published on HuggingFace and Kaggle.

---

## Tech stack

- **Model**: PyTorch + EfficientNet-B0 (transfer learning)
- **Visualization**: Grad-CAM (custom implementation, no extra library needed)
- **App**: Streamlit → deployed on HuggingFace Spaces
- **Training**: Google Colab (free T4 GPU)
- **Data**: Mendeley, HuggingFace, Kaggle + original community photos

---

*Built by Sahas · Denmark High School · Alpharetta, GA*
