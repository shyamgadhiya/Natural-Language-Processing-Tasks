from transformers import MarianMTModel, MarianTokenizer
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
batch = tokenizer(["Machine translation is powerful.we have to be careful about ai because artificial intelligence is now entireing in artificial general intelligence."],
return_tensors="pt", padding=True)
translated = model.generate(**batch)
print(tokenizer.decode(translated[0], skip_special_tokens=True))