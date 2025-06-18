# AI-Powered Email Categorizer & Reply Generator

## Project Overview

This project classifies email intents using DistilBERT and generates replies using transformer models. Dataset used is Enron Email Dataset.

## Folder Structure

- `data/` - Raw and labeled datasets
- `notebooks/` - Preprocessing and training notebooks
- `models/` - Saved models
- `app/` - Backend API (to be created)
- `frontend/` - Streamlit app (optional)
- `mlops/` - Docker & deployment files

## How to Run

1. Download Enron dataset and place in `data/raw_enron.csv`
2. Run preprocessing notebook: `notebooks/1_preprocessing.ipynb`
3. Run classifier training notebook: `notebooks/2_classifier_training.ipynb`
4. Next: build reply generator and API (forthcoming)

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- FastAPI (later)
- Streamlit (optional)

## Author

Pubudu Weerasinghe
