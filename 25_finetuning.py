import os
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 1. Use DistilBERT instead of full BERT (Much faster on CPU)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Load dataset
dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")


def tokenize_function(examples):
  return tokenizer(examples["text"], truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# (Optional) For quick experimentation, use a smaller split:
train_ds = tokenized_datasets["train"].shuffle(seed=42).select(range(2000))
val_ds = tokenized_datasets["validation"].select(range(500))

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 3. Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

# 4. Metrics
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
  logits, labels = eval_pred
  preds = np.argmax(logits, axis=-1)
  return accuracy.compute(predictions=preds, references=labels)


# 5. Training Arguments tuned for CPU
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  # 1 epoch to verify everything works fast
    per_device_train_batch_size=32,  # Larger batch = far fewer steps (63 steps instead of 3201)
    per_device_eval_batch_size=64,
    learning_rate=3e-5,
    eval_strategy="steps",
    eval_steps=30,
    logging_steps=10,
    dataloader_pin_memory=False,  # Suppresses CPU pin_memory warning
    use_cpu=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nStarting fast training...")
trainer.train()

# Save model, config, and tokenizer artifacts
save_path = "./my_finetuned_sentiment_model"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model and tokenizer saved cleanly to: {save_path}")