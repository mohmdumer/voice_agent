# Week 2 Write-up — Multimodal Model Development (Baseline)

## Goal

Establish a captioning baseline on Flickr8k using pretrained BLIP: first zero-shot, then lightly
fine-tuned, tracked end-to-end in MLflow so every run is comparable.

## Model and approach

- **Model:** `Salesforce/blip-image-captioning-base` (Hugging Face)
- **Zero-shot baseline:** no training, straight inference
- **Fine-tuning:** vision encoder frozen, text decoder fine-tuned on ~3,000 Flickr8k caption pairs
  (1,500 images), 1 epoch — done on **Kaggle GPU** (T4), since local dev machine is CPU-only by
  design (see Week 1 write-up)

## Results

All metrics computed with sacrebleu (BLEU) and rouge-score (ROUGE-1/2/L) against Flickr8k's human
reference captions. Full detail and MLflow logging in `notebooks/week2_day1..day4_*.ipynb`.

| Run | Fine-tuned | Decoding | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|---|---|
| Zero-shot baseline | No | greedy | 16.7 | 50.2 | 26.6 | 48.1 |
| A: lr=5e-5 | Yes | greedy | 27.2 | 54.0 | 32.2 | 50.5 |
| B: lr=1e-5 | Yes | greedy | 24.8 | 52.8 | 30.0 | 49.6 |
| **C: lr=5e-5** | **Yes** | **beam search** | **28.9** | **55.3** | **33.6** | **52.2** |

*(Zero-shot scored on a 300-image subset; fine-tuned configs on a 100-image held-out subset —
not perfectly apples-to-apples in sample size, but the gap is far larger than sampling noise
would explain.)*

**Winner: config C** — BLIP fine-tuned at lr=5e-5 (vision encoder frozen), captioned with beam
search (num_beams=4) instead of greedy decoding at inference. It leads on all four metrics
simultaneously, and reuses config A's exact weights — the beam-search gain is essentially free
(a little extra inference latency, no extra training).

## What we learned

- **Even light fine-tuning helps a lot.** Freezing the vision encoder and tuning only the text
  decoder on a small subset (1,500 images, 1 epoch) lifted BLEU by 48-73% relative over zero-shot.
  Flickr8k's captions have a fairly consistent style/vocabulary that the base model doesn't fully
  match zero-shot; a light fine-tune closes most of that gap cheaply.
- **Decoding strategy matters, and it's nearly free.** Switching config A's inference from greedy
  to beam search (config C) improved every metric with no retraining — worth defaulting to beam
  search for the final model.
- **A lower learning rate wasn't better here.** Config B (lr=1e-5) underperformed A/C. Its
  `final_train_loss` was numerically lower, but that value is the *last mini-batch's* loss (noisy,
  not an epoch average) — not a reliable signal on its own. With only 1,500 images and 1 epoch,
  the lower LR likely just adapted less to the domain within the available steps.
- **Qualitative check (Day 1) matters alongside the numbers.** Zero-shot BLIP got the general
  subject/action right most of the time but was terser than human captions and had at least one
  clear miscategorization (a puppy captioned as "a small pig") — a good candidate for Week 3's
  Grad-CAM analysis.

## Infrastructure notes

- MLflow tracking backend switched from the plain filesystem store to SQLite
  (`sqlite:///mlflow.db`) — MLflow 3.x deprecated the file store for new projects.
- Kaggle GPU notebooks are the fine-tuning workflow for the rest of the internship: train there,
  download `results/*.json` + `.csv` (small, kept in git) and any checkpoints, then log into the
  local MLflow store. `notebooks/kaggle_week2_finetune_blip.ipynb` is reusable for future
  experiments — just adjust the configs.

## Next steps (Week 3 preview)

Curate 10-15 examples (mix of correct/incorrect captions, including the "puppy → pig" case) and
run Grad-CAM (Captum) over the fine-tuned model's vision encoder, plus SHAP/LIME on the decoder
side, to understand what the model attends to when it gets captions right vs. wrong.
