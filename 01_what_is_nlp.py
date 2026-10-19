import nltk
import spacy

print("NLTK version:", nltk.__version__)
nlp = spacy.load("en_core_web_sm")
doc = nlp("Language models learn patterns from text and I learn from Language models.")
print("Tokens:", [t.text for t in doc])