from transformers import pipeline
ner = pipeline("ner", aggregation_strategy="simple")
text = "Satya Nadella leads Microsoft, based in Redmond"
for entity in ner(text):
    print(entity["word"], "->", entity["entity_group"])