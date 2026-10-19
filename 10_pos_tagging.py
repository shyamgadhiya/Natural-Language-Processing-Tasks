import nltk
text = "The quick brown fox jumps over the lazy dog"
tokens = nltk.word_tokenize(text)
tags = nltk.pos_tag(tokens)
nouns = [w for w, t in tags if t.startswith("NN")]
verbs = [w for w, t in tags if t.startswith("VB")]
print("Nouns:", nouns, "Verbs:", verbs)
