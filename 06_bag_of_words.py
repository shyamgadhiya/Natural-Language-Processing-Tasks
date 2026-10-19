from sklearn.feature_extraction.text import CountVectorizer

corpus = ["I love NLP", "NLP loves data", "data drives NLP"]
vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(corpus)
print("Vocabulary:", vectorizer.get_feature_names_out())
print("Matrix shape:", matrix.shape)
print(matrix.toarray())