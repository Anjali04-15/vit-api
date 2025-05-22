# vit-api
FastAPI service for ViT model

# ViT FastAPI Model Deployment

This repository contains a FastAPI backend to serve a Vision Transformer (ViT) model for fruit classification. The API accepts images and returns the predicted class.

## Project Structure

- `main.py` — FastAPI app entry point
- `vit_model.py` — Vision Transformer model and inference code
- `vit_weights.pt` — Pretrained ViT model weights
- `requirements.txt` — Python dependencies
- `.gitignore` — Files and folders to ignore in Git

## Setup Instructions

1. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Run the FastAPI server:
    ```bash
    uvicorn main:app --reload

4. Access API docs at: 
    ```bash 
    http:localhost:8000/docs