# app/model_intent.py

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import numpy as np

class IntentClassifier:
    def __init__(self, model_path='../models/intent_classifier'):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.labels = ['meeting_request', 'complaint', 'social', 'task_update', 'general']  # Use same order as training

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred_idx = np.argmax(probs)
        return {
            "label": self.labels[pred_idx],
            "confidence": float(probs[0][pred_idx])
        }
