from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
text = "The service was okay, but the wait time was way too long."
vader = SentimentIntensityAnalyzer()
print("VADER:", vader.polarity_scores(text)["compound"])
transformer = pipeline("sentiment-analysis")
result = transformer(text)[0]
print("Transformer:", result["label"], round(result["score"], 2))
