from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
texts = ["win a free prize now", "meeting at 3pm tomorrow",
"claim your free reward", "project deadline is friday"]
labels = ["spam", "ham", "spam", "ham"]
vec = TfidfVectorizer()
X = vec.fit_transform(texts)
clf = MultinomialNB().fit(X, labels)
new = vec.transform(["free meeting reward tomorrow"])
print("Prediction:", clf.predict(new)[0])