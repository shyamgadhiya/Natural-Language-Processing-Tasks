from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from collections import Counter
text = "NLP is field of AI, AI is future ,NLP is also field of future"
tokens = [w.lower() for w in word_tokenize(text) if w.isalnum()]
bigrams = list(ngrams(tokens, 2))
counts = Counter(bigrams)
print("Top 3 most frequent bigrams:")
print(counts.most_common(3))