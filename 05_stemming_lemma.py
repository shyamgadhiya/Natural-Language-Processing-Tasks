from nltk.stem import PorterStemmer, WordNetLemmatizer
#stemming just cut the from words s and es and ing they don't care about the grammar or structure of the word
#lemma it consider the grammar and structure of the word

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
words = ["running", "flies", "better", "studies", "brightness", "swimming", "was"]
for w in words:
    print(w, "->", stemmer.stem(w), "/", lemmatizer.lemmatize(w, pos="v"))