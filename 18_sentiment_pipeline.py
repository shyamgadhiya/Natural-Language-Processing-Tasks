from transformers import pipeline
classifier = pipeline("sentiment-analysis")
texts = ["I love this product!", "This was a waste of time.",
"It's okay, nothing special."]
for text in texts:
    result = classifier(text)[0]
    print(text, "->", result["label"], round(result["score"], 2))