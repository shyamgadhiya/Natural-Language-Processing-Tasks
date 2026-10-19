from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

app = FastAPI(title="Sentiment Analysis API")

# Explicit model definition to avoid deprecation warnings
classifier = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)


class TextInput(BaseModel):
  text: str = Field(
      ...,
      max_length=1000,
      description="Input text must not exceed 1000 characters.",
  )


@app.post("/analyze")
async def analyze(data: TextInput):
  result = classifier(data.text)[0]
  return {"label": result["label"], "score": float(result["score"])}