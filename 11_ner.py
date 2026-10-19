import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple was founded by Steve Jobs in Cupertino")
for ent in doc.ents:
    print(ent.text, "->", ent.label_)