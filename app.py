from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import torch
import random

app = FastAPI(title="AI Sentiment Chatbot")

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Use GPU if available
device = 0 if torch.cuda.is_available() else -1

# Load BERT model
sentiment_pipeline = pipeline(
    "text-classification",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=device
)

# Suicidal keywords (priority check)
suicidal_keywords = [
    "kill myself", "end my life", "want to die",
    "suicidal", "i can't go on", "no reason to live"
]

# Responses
intent_responses = {
    "positive": [
        "That's wonderful to hear! 😊",
        "Love that energy! Keep going! 💪"
    ],
    "neutral": [
        "Got it 👍",
        "Thanks for sharing."
    ],
    "negative": [
        "I'm sorry you're feeling this way. 😔",
        "Things can improve. I'm here with you."
    ],
    "suicidal": [
        "I’m really concerned about you. Please reach out to someone immediately. 💛",
        "You matter. If you’re in India call KIRAN: 1800-599-0019.",
        "If you're in the US call 988. In the UK call Samaritans: 116 123."
    ]
}

# Request format
class TextRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict_sentiment(request: TextRequest):

    user_text = request.text.lower()

    # 1️⃣ Check suicidal first
    if any(word in user_text for word in suicidal_keywords):
        return {
            "sentiment": "Suicidal",
            "confidence": "High",
            "response": random.choice(intent_responses["suicidal"])
        }

    # 2️⃣ BERT Sentiment
    result = sentiment_pipeline(request.text)
    label = result[0]["label"]
    score = round(result[0]["score"], 2)

    if "5 stars" in label or "4 stars" in label:
        sentiment = "positive"
    elif "3 stars" in label:
        sentiment = "neutral"
    else:
        sentiment = "negative"

    return {
        "sentiment": sentiment.capitalize(),
        "confidence": score,
        "response": random.choice(intent_responses[sentiment])
    }
