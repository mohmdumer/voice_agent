# voice_agent — Multimodal NLP + Explainable AI

Image captioning on the [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset, built out from a notebook into a tracked, explainable, containerized, CI/CD-deployed service. Internship project following `Internship Plan - Multimodal NLP and XAI.docx`.

## Goal

- Caption images with a pretrained multimodal model (BLIP), lightly fine-tuned
- Explain predictions with Grad-CAM / SHAP / LIME (Captum, XAI)
- Track experiments with MLflow, version data/models with DVC
- Serve via FastAPI, containerize with Docker, deploy with CI/CD (GitHub Actions) to Hugging Face Spaces / Render
- Demo UI in Gradio/Streamlit

## Project layout

```
data/            Flickr8k images + captions (gitignored; not committed)
notebooks/       exploration / preprocessing / experiment notebooks
src/             modular pipeline code (data_loader, model, inference, xai)
tests/           pytest unit tests
app/             FastAPI service
```

## Environment setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Training/fine-tuning runs on **Kaggle GPU notebooks**; local `.venv` is CPU-only for development, preprocessing, and inference.

## Status

Week 1 (data exploration + preprocessing) in progress — see project plan for the full 5-week roadmap.
