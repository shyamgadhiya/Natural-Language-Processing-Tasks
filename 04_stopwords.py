from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words("english"))
text = "This is a simple example showing stopword removal"
tokens = word_tokenize(text)
filtered = [w for w in tokens if w.lower() not in stop_words]
print("Before:", len(tokens), "After:", len(filtered))
print(filtered)

