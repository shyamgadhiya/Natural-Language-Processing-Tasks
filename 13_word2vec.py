from gensim.models import Word2Vec
sentences = [["nlp", "is", "fun"], ["deep", "learning", "powers", "nlp"],
 ["word", "vectors", "capture", "meaning"]]
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1)
print(model.wv.most_similar("learning", topn=3))