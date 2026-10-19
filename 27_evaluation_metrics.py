from nltk.translate.bleu_score import sentence_bleu
reference = [["the", "cat", "sat", "on", "the", "mat"]]
candidate = ["the", "cat", "is", "on", "the", "mat"]
bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))

# BLEU-2 (Bigram match)
bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))

# BLEU-3 (Trigram match)
bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))

print(f"BLEU-1: {round(bleu1, 3)}")  # 0.833
print(f"BLEU-2: {round(bleu2, 3)}")  # 0.745
print(f"BLEU-3: {round(bleu3, 3)}")  # 0.605