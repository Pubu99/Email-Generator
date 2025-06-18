# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from app.model_intent import IntentClassifier
from app.model_reply import ReplyGenerator
from app.ner import extract_entities

app = FastAPI(
    title="AI-Powered Email Categorizer & Reply Generator",
    description="API with dynamic entity extraction and personalized reply generation.",
    version="1.1"
)

intent_classifier = IntentClassifier()
reply_generator = ReplyGenerator()

class EmailInput(BaseModel):
    email_text: str

class EmailWithCategory(BaseModel):
    email_text: str
    category: str

class EmailWithCategoryAndEntities(BaseModel):
    email_text: str
    category: str
    entities: dict = None  # Optional: user can override extracted entities

@app.post("/classify")
def classify_email(input: EmailInput):
    result = intent_classifier.predict(input.email_text)
    return {"intent": result["label"], "confidence": result["confidence"]}

@app.post("/extract_entities")
def extract_email_entities(input: EmailInput):
    entities = extract_entities(input.email_text)
    return {"entities": entities}

@app.post("/generate")
def generate_reply(input: EmailWithCategoryAndEntities):
    # If entities not provided by user, extract automatically
    entities = input.entities if input.entities else extract_entities(input.email_text)
    reply = reply_generator.generate(input.email_text, input.category, entities)
    return {"reply": reply, "used_entities": entities}
