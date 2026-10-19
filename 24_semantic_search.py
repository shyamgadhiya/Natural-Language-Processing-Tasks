from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")
docs = ["A dog barks in the park", "Stock markets rose today",
"Puppies playing outside"]
query = "animals playing outdoors"
doc_emb = model.encode(docs, convert_to_tensor=True)
q_emb = model.encode(query, convert_to_tensor=True)
scores = util.cos_sim(q_emb, doc_emb)[0]
ranked = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)
for doc, score in ranked:
    print(round(score, 3), doc)