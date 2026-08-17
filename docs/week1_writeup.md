# Week 1 Write-up — Foundations, Environment & Data Exploration

## Problem statement

Build an image-captioning system on the Flickr8k dataset: given a photo, generate a natural-language
description of its contents. This is the base task for the rest of the internship, which layers
explainability (Week 3), experiment tracking and packaging (Weeks 2 & 4), and a deployed, CI/CD-driven
service (Week 5) on top of it.

## Dataset summary

- **Source:** Flickr8k (images + `captions.txt`)
- **Size:** 8,091 images, 40,455 captions (5 captions per image, uniformly — confirmed, no images with
  more or fewer)
- **Caption length:** mean 11.8 words, median 11, min 1, max 38 (see histogram in
  `notebooks/day2_data_exploration.ipynb`)
- **Vocabulary:** 8,904 unique tokens (lowercased, punctuation stripped, stop words kept); 8,788 after
  stop-word removal
- **Most common words:** dominated by generic nouns/adjectives typical of captioning datasets — "dog",
  "man", "two", "white", "black", "boy", "woman", "girl", "wearing", "people"

## Data quality findings

- **Duplicate captions:** 20 exact `(image, caption)` duplicate rows out of 40,455 — negligible, not
  addressed (kept as-is; duplicates don't distort captioning training meaningfully at this scale)
- **Short/blank captions:** 17 captions under 3 words (e.g., "dogs racing", "A", "man surfing"). Left
  in the dataset for now; may be filtered before fine-tuning in Week 2 if they hurt loss stability
- **Corrupted/unreadable images:** 0 out of 8,091 — every image was readable by OpenCV with a consistent
  3-channel shape

## Preprocessing decisions

- **Text:** lowercase → NLTK tokenize → strip punctuation and English stop words → build vocabulary
  (`notebooks/notebook_EDA.ipynb`). Stop-word removal trades some fluency signal for a smaller, more
  content-focused vocabulary; this is fine for later analysis (e.g., XAI token attribution in Week 3)
  but the raw (non-stop-word-stripped) captions are kept as ground truth for BLEU/ROUGE scoring in
  Week 2, since those metrics expect natural sentences.
- **Images:** read with OpenCV (BGR→RGB), resized to 224×224, converted to tensor, normalized with
  ImageNet mean/std (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`) — matches the input convention
  expected by the pretrained BLIP vision encoder used in Week 2, so no re-normalization is needed at
  fine-tuning time. All 8,091 processed tensors are saved to `data/processed/images/*.pt`.

## Environment

- `.venv` recreated on **Python 3.12** (the original 3.9 environment had no Windows wheels for `spacy`,
  since its `thinc` dependency dropped 3.9 support)
- `requirements.txt` covers all Week 1 dependencies (nltk, spacy, transformers, datasets, opencv-python,
  torchvision, torch, jupyter, tqdm, requests, matplotlib, pandas, pillow) and is committed for
  reproducibility
- **Compute split:** local machine (this `.venv`, CPU-only) for all preprocessing/dev/inference work;
  **Kaggle GPU notebooks** for any fine-tuning or other GPU-heavy runs starting Week 2

## Repo structure

```
data/            Flickr8k images + captions (gitignored, not committed — will be DVC-tracked in Week 4)
notebooks/       exploration and preprocessing notebooks
src/, tests/, app/   scaffolded, to be filled in from Week 2 onward
```

## Next steps (Week 2 preview)

Run zero-shot BLIP inference on sample images, evaluate with BLEU/ROUGE on a validation subset, set up
MLflow tracking, and run a few light fine-tuning experiments on Kaggle to pick a baseline model.
