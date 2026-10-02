# Foundations to Frontiers: Multi-Task Deep Learning & NLP in PyTorch

This repository contains a comprehensive "Neural Network Decathlon" highlighting a progression of deep learning techniques applied to various Natural Language Processing (NLP) and classification tasks. Built primarily in **PyTorch**, this project demonstrates the ability to construct architectures from scratch, tune deep neural networks, and leverage state-of-the-art pre-trained Transformer models.

## 🏆 The Neural Network Decathlon: 5 Projects in 1
To provide a quick understanding of the repository's scope, this notebook is divided into five distinct machine learning projects, escalating in complexity from traditional deep learning to modern language modeling:

1. **Multiclass Text Classification:** Custom PyTorch MLP with TF-IDF and a massive 162-combination hyperparameter grid search.
2. **Binary Spam Detection:** Deep Neural Network calibrated with BCEWithLogitsLoss for high-accuracy spam filtering.
3. **Transformer Fine-Tuning:** Transfer learning using Hugging Face's `DistilBERT` for real-world disaster tweet classification.
4. **Word Embeddings Exploration:** Custom `Word2Vec` training from scratch using Gensim, paired with a Random Forest classifier.
5. **Bengio et al. (2003) Implementation:** A ground-up PyTorch replication of the foundational Neural Probabilistic Language Model.

---

## 📊 Project Details & Results

### Task 1: 20 Newsgroups Topic Classification
* **Objective:** Classify text documents into distinct newsgroup categories (e.g., sci.space, comp.graphics).
* **Architecture:** `TunedTextClassifier` — A PyTorch neural network featuring configurable hidden dimensions, Dropout, and Batch Normalization.
* **Techniques:** 
  * Custom `scikit-learn` preprocessing pipeline (Text cleaning, Lemmatization, Stopword removal).
  * TF-IDF Vectorization tested across varying N-gram ranges `(1,1), (1,2), (1,3)`.
  * Rigorous hyperparameter tuning evaluating **162 distinct combinations** (Learning Rate, Batch Size, Hidden Dims, Dropout Rates, Weight Decay).
* **Results:** Achieved **91.15% validation accuracy** utilizing optimal parameters (`lr=0.001`, `batch_size=32`, `hidden_dims=[512, 256]`, `dropout=[0.6, 0.4]`, `weight_decay=0.0001`).

### Task 2: Binary Spam Detection
* **Objective:** Identify spam emails using the UCI Spambase dataset.
* **Architecture:** `SpamClassifier` — A PyTorch MLP outputting raw logits.
* **Techniques:** 
  * 80/10/10 Train/Validation/Test split with `StandardScaler` feature normalization.
  * Custom thresholding logic and `BCEWithLogitsLoss` for robust binary classification.
  * Learning rate scheduling (`ReduceLROnPlateau`) to combat overfitting.
* **Results:** Achieved **94.36% test accuracy**, with an F1-score of 0.95 for Non-Spam and 0.93 for Spam.

### Task 3: Disaster Tweet Classification with DistilBERT
* **Objective:** Classify tweets as relating to real disaster events or not.
* **Architecture:** `DistilBertForSequenceClassification` (Pre-trained Transformer).
* **Techniques:**
  * Integration with Hugging Face `transformers` and `datasets` libraries.
  * Tokenization and padding/truncation to a max length of 128.
  * Fine-tuning a 66M+ parameter model over 3 epochs using `AdamW`.
* **Results:** Reached **83% accuracy** on the held-out validation set.

### Task 4: Exploring Word Vectors (Word2Vec)
* **Objective:** Train and analyze word embeddings from scratch using the disaster tweets dataset.
* **Architecture:** `Word2Vec` (Gensim Skip-Gram) + `RandomForestClassifier`.
* **Techniques:**
  * Custom regex-based tweet cleaner mapping raw text to tokenized sentences.
  * Generation of averaged feature vectors for entire tweets based on learned embeddings.
* **Results:** 
  * The model successfully learned semantic relationships (e.g., `fire` showed >0.95 similarity to `truck` and `forest`).
  * The downstream Random Forest classifier achieved **75% validation accuracy** based purely on the custom embeddings.

### Task 5: Neural Probabilistic Language Model (Bengio 2003)
* **Objective:** Replicate the pioneering 2003 neural language model to predict the next word in a sequence based on a context window.
* **Architecture:** Custom PyTorch model featuring a shared Embedding table (C matrix), hidden `tanh` layers, and optional direct skip-connections from the embedding layer to the output layer.
* **Techniques:**
  * Custom `torch.utils.data.Dataset` mapping the NLTK Brown Corpus into `(context, next_word)` tensor pairs.
  * Implementation of weight decay (L2 regularization) to stabilize severe overfitting common in early language models.
* **Results:** Achieved a **Test Perplexity of 901.80** and a Validation Perplexity of 871.09, demonstrating effective generalization on the 56,000+ word vocabulary of the Brown Corpus.

---

## 🛠️ Technologies & Frameworks
* **Deep Learning:** PyTorch (`torch.nn`, `torch.optim`, `DataLoader`, `TensorDataset`)
* **NLP & Transformers:** Hugging Face (`transformers`, `datasets`), NLTK, Gensim (`Word2Vec`)
* **Machine Learning:** Scikit-Learn (Pipelines, Random Forest, Metrics)
* **Data Processing & Visualization:** Pandas, NumPy, Matplotlib, tqdm

## 🚀 Getting Started

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install torch transformers datasets gensim scikit-learn nltk pandas matplotlib torchinfo
