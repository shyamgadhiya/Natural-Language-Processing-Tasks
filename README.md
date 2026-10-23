# Natural Language Processing (NLP) Tasks & Roadmap

A step-by-step hands-on guide and reference codebase covering NLP from foundational preprocessing and classical machine learning to modern Transformer architectures, fine-tuning, and API deployment.

---

## Repository Structure & Modules

The repository is organized progressively into distinct learning phases:

### 1. Fundamentals & Preprocessing
* **`01_what_is_nlp.py`**: Quick introduction to token processing with NLTK and spaCy.
* **`02_preprocessing.py`**: Text normalization, lowercasing, and punctuation cleanup.
* **`03_tokenization.py`**: Word and sentence segmentation.
* **`04_stopwords.py`**: Filtering domain and general stop words.
* **`05_stemming_lemma.py`**: Differences between rule-based stemming and vocabulary-aware lemmatization.
* **`08_ngrams.py`**: Generating contiguous sequences (unigrams, bigrams, trigrams).
* **`09_regex_extract.py`**: Pattern extraction using regular expressions.

### 2. Traditional Feature Extraction & Classical ML
* **`06_bag_of_words.py`**: Text vectorization using CountVectorizer (Bag of Words).
* **`07_tfidf.py`**: TF-IDF scoring and cosine similarity calculation.
* **`10_pos_tagging.py`**: Part-of-speech (POS) tagging with grammatical annotations.
* **`11_ner.py`**: Named Entity Recognition (NER) with spaCy.
* **`12_naive_bayes.py`**: Text classification with Multinomial Naive Bayes.

### 3. Embeddings & Deep Learning Foundations
* **`13_word2vec.py`**: Continuous word representations using Gensim Word2Vec.
* **`14_lstm_shapes.py`**: PyTorch LSTM tensor input/output shape breakdown.
* **`15_attention_scores.py`**: Implementing scaled dot-product self-attention mechanism.
* **`16_transformer_overview.py`**: Core architecture and components of the Transformer model.
* **`17_bert_hidden_state.py`**: Extracting contextual token embeddings from BERT hidden states.

### 4. Hugging Face Pipelines & Applications
* **`18_sentiment_pipeline.py`**: Out-of-the-box sentiment classification.
* **`19_ner_transformers.py`**: Transformer-based entity recognition pipeline.
* **`20_translation.py`**: Sequence-to-sequence neural machine translation.
* **`21_summarization.py`**: Abstractive text summarization.
* **`22_que_ans.py`**: Extractive question answering over context passages.
* **`23_sentiment_compare.py`**: Comparing multiple sentiment model outputs and behaviors.
* **`24_semantic_search.py`**: Sentence embeddings and cosine similarity search using Sentence-Transformers.

### 5. Fine-Tuning & Evaluation
* **`25_finetuning.py`**: End-to-end model fine-tuning with Hugging Face `Trainer` API.
* **`26_test_model.py`**: Inference and qualitative validation on the fine-tuned model checkpoint.
* **`27_evaluation_metrics.py`**: Standard NLP classification metrics (Accuracy, Precision, Recall, F1).

### 6. Deployment & Benchmarking
* **`28_app.py`**: Production-ready REST API using FastAPI and Pydantic validation for sentiment analysis.
* **`pipeline.py`**: Batch inference throughput and dataset streaming performance benchmark.

---

## Getting Started

### Prerequisites
- Python 3.9+ installed
- CUDA-compatible GPU (recommended for fine-tuning scripts `25_finetuning.py`, but CPU works for inference scripts)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shyamgadhiya/Natural-Language-Processing-Tasks.git
   cd Natural-Language-Processing-Tasks
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required language models:**
   ```bash
   # spaCy English small model
   python -m spacy download en_core_web_sm

   # NLTK resources (if prompted by scripts)
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

---

## Running the Examples

### Running Individual Scripts
Execute any script directly with Python:
```bash
python 07_tfidf.py
python 18_sentiment_pipeline.py
python 24_semantic_search.py
```

### Running the FastAPI Application
Start the sentiment analysis web server:
```bash
uvicorn 28_app:app --reload --port 8000
```
- Interactive Swagger docs will be available at: `http://127.0.0.1:8000/docs`
- Send a sample request:
  ```bash
  curl -X POST "http://127.0.0.1:8000/analyze" \
       -H "Content-Type: application/json" \
       -d '{"text": "Natural Language Processing with Transformers is amazing!"}'
  ```
