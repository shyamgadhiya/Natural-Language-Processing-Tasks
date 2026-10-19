from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer("NLP models understand context", return_tensors="pt")
outputs = model(**inputs)
print("Last hidden state:", outputs.last_hidden_state.shape)
