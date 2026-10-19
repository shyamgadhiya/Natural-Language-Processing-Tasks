from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["The apple, scientifically known as Malus domestica, is one of the most widely cultivated and consumed fruits in the world. Originating in Central Asia, where its wild ancestor Malus sieversii can still be found today, the apple has thousands of varieties, ranging from the sweet Fuji and Gala to the crisp Honeycrisp and tart Granny Smith. Beyond its crisp texture and refreshing taste, the apple is highly nutritious. It is packed with dietary fiber (pectin), vitamin C, and various antioxidants that support heart health, aid digestion, and help regulate blood sugar levels, truly justifying the age-old proverb,An apple a day keeps the doctor away."]
tfidf = TfidfVectorizer()
scores = tfidf.fit_transform(corpus)
terms = tfidf.get_feature_names_out()
for doc_id in range(len(corpus)):
    row = scores[doc_id].toarray()[0]
    print(f"Doc {doc_id} top term:", terms[row.argmax()])