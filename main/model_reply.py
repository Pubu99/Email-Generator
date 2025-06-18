# app/model_reply.py

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class ReplyGenerator:
    def __init__(self, model_path='../models/reply_generator'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()

    def create_prompt(self, email_text, category, entities):
        # Use first found entity or empty string
        person = entities.get("PERSON", [""])[0] if entities.get("PERSON") else ""
        date = entities.get("DATE", [""])[0] if entities.get("DATE") else ""
        time = entities.get("TIME", [""])[0] if entities.get("TIME") else ""
        location = entities.get("GPE", [""])[0] if entities.get("GPE") else ""

        prompt = (
            f"Email: {email_text}\n"
            f"Category: {category}\n"
            f"Person: {person}\n"
            f"Date: {date}\n"
            f"Time: {time}\n"
            f"Location: {location}\n"
            f"Generate a polite and context-aware reply."
        )
        return prompt

    def generate(self, email_text, category, entities, max_length=80):
        prompt = self.create_prompt(email_text, category, entities)
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=256, truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                top_k=50,
                temperature=0.7
            )
        reply = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return reply
