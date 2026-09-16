# 💬 Text Emotion Classification (NLP)

An NLP project that classifies short text statements into emotions (sadness, anger, love, joy, etc.) using classic text preprocessing and vectorization techniques, comparing **Naive Bayes** and **Logistic Regression** across two different feature representations — **Bag of Words** and **TF-IDF**.

## 📊 Dataset

- Text-emotion dataset (`train.txt`, semicolon-separated) with two columns: raw text and an emotion label
- Example: `"i didnt feel humiliated"` → `sadness`
- Emotion labels were label-encoded into integers before modeling

## 🛠️ Workflow

1. **Import Libraries** — NumPy, Pandas, Matplotlib, Seaborn
2. **Data Gathering** — loaded from `train.txt` (Kaggle emotion dataset)
3. **Text Cleaning**
   - Encoded emotion labels to integers
   - **Lowercasing** all text
   - **Punctuation removal**
   - **Number removal**
   - **URL/link removal**
   - **Emoji/non-ASCII character removal**
   - **Stopword removal** using NLTK's English stopword list (198 words)
4. **Train-Test Split** — 67/33 split (`random_state=42`)
5. **Vectorization** — text converted to numeric features two ways:
   - **Bag of Words** (`CountVectorizer`) — 9,482 features
   - **TF-IDF** (`TfidfVectorizer`)
6. **Model Training** — classifiers trained and evaluated on each representation:
   - Naive Bayes (Bag of Words)
   - Naive Bayes (TF-IDF)
   - Logistic Regression (TF-IDF)

## 📈 Results

| Model | Feature Representation | Accuracy |
|---|---|---|
| Naive Bayes | Bag of Words | 72.8% |
| Naive Bayes | TF-IDF | 63.4% |
| **Logistic Regression** | **TF-IDF** | **81.2%** |

> Logistic Regression on TF-IDF features gives the best result by a clear margin. Interestingly, Naive Bayes performs *worse* with TF-IDF than with raw Bag-of-Words counts — Multinomial Naive Bayes assumes count-like (integer) feature distributions, so it's naturally suited to BoW, while TF-IDF's continuous weighted values work against that assumption. This makes Logistic Regression + TF-IDF the strongest combination here.

## 🧰 Tech Stack

- **Python 3**
- **Pandas** & **NumPy** — data handling
- **Matplotlib** & **Seaborn** — visualization
- **NLTK** — stopword removal
- **Scikit-learn** — vectorization (`CountVectorizer`, `TfidfVectorizer`), models, and metrics

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk
```

The notebook downloads required NLTK resources automatically:
```python
nltk.download('punkt')
nltk.download('stopwords')
```

### Data

Place `train.txt` (semicolon-separated `text;emotion` format) in the working directory, or update the file path in the notebook.

### Run

```bash
jupyter notebook NLP_Project-1.ipynb
```

Run all cells sequentially to reproduce preprocessing and model results.

## 📁 Project Structure

```
.
├── NLP_Project-1.ipynb   # Main notebook: text cleaning, vectorization, model training & comparison
└── README.md
```

## 🔮 Future Improvements

- Add precision/recall/F1 and confusion matrix per emotion class (accuracy alone hides class-level performance, especially with imbalanced emotions)
- Try stemming/lemmatization on top of stopword removal
- Experiment with n-grams (bigrams/trigrams) in vectorization
- Try word embeddings (Word2Vec, GloVe) or a fine-tuned transformer (e.g. DistilBERT) for richer text representations
- Hyperparameter tuning (e.g. `alpha` for Naive Bayes, `C` for Logistic Regression)

## 👤 Author

**Muhammad Ibrahim**

- 📧 Email: [mibrahim.seng@gmail.com](mailto:mibrahim.seng@gmail.com)
- 💼 LinkedIn: [muhammad-ibrahim-python](https://www.linkedin.com/in/muhammad-ibrahim-python)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
