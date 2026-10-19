from nltk.tokenize import word_tokenize, sent_tokenize
text = "NLP is fun. Tokenizing text is the first step isn't it?"
sentences = sent_tokenize(text)
words = word_tokenize(text)
print("Sentences:", len(sentences))
print("Words:", len(words))