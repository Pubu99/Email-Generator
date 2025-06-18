# app/model_reply.py

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class ReplyGenerator:
    def __init__(self, model_path='../models/reply_generator'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()

    def generate(self, email_text, category, max_length=64):
        input_text = f"Email: {email_text} Category: {category}"
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", max_length=128, truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
        reply = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return reply
