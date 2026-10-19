import os
import torch
from transformers import pipeline

# Suppress Windows symlink warning if present
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 1. Load fine-tuned model
finetuned_classifier = pipeline(
    "text-classification",
    model="./my_finetuned_sentiment_model",
    device=-1,  # CPU
)

# 2. Load the original base model (before fine-tuning)
base_classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased",
    device=-1,  # CPU
)

test_sentences = [
    "The screenplay was brilliant and the acting felt completely natural.",
    "A dull, predictable plot that dragged on far too long.",
    "Surprisingly entertaining despite some minor flaws in the pacing.",
    "I wanted to love this, but the ending ruined everything.",
]

# Run predictions across both models
finetuned_results = finetuned_classifier(test_sentences)
base_results = base_classifier(test_sentences)

# Label mapping for raw logits (0 -> Negative, 1 -> Positive)
label_map = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "POSITIVE",
    "negative": "NEGATIVE",
    "positive": "POSITIVE",
}

print(f"{'=' * 80}")
print(f"{'TEXT':<50} | {'BASE MODEL':<12} | {'FINE-TUNED':<12}")
print(f"{'=' * 80}")

for text, base_res, ft_res in zip(test_sentences, base_results, finetuned_results):
  base_label = label_map.get(base_res["label"], base_res["label"])
  base_conf = base_res["score"] * 100

  ft_label = label_map.get(ft_res["label"], ft_res["label"])
  ft_conf = ft_res["score"] * 100

  # Shorten text if needed for clean display
  truncated_text = text if len(text) <= 47 else text[:44] + "..."

  print(
      f"{truncated_text:<50} | {base_label} {base_conf:5.1f}% | {ft_label} {ft_conf:5.1f}%"
  )

print(f"{'=' * 80}")