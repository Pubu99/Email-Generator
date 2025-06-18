from fastapi import FastAPI
from pydantic import BaseModel
from app.model_intent import IntentClassifier
from app.model_reply import ReplyGenerator

app = FastAPI(
    title="AI-Powered Email Categorizer & Reply Generator",
    description="API for classifying email intents and generating replies.",
    version="1.0"
)

intent_classifier = IntentClassifier()
reply_generator = ReplyGenerator()

class EmailInput(BaseModel):
    email_text: str

class EmailWithCategory(EmailInput):
    category: str

@app.post("/classify")
def classify_email(input: EmailInput):
    result = intent_classifier.predict(input.email_text)
    return {"intent": result["label"], "confidence": result["confidence"]}

@app.post("/generate")
def generate_reply(input: EmailWithCategory):
    reply = reply_generator.generate(input.email_text, input.category)
    return {"reply": reply}
