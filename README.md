# fakenewsdetection

---

# 📰 Fake News Detection AI

A simple **Machine Learning project** that detects whether a given news article is **FAKE** or **REAL**.
The model uses **TF-IDF Vectorization** and a **Passive Aggressive Classifier** to classify news text.

---

## 📌 Features

* Classifies news as **FAKE** or **REAL**
* Trains on the **Fake & Real News Dataset** from Kaggle
* Provides accuracy, confusion matrix, and classification report
* Interactive CLI mode → enter your own news text and get predictions

---

## 📊 Dataset

We use the [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

You can use either:

1. `dataset.csv` → with columns `text,label`
2. `True.csv` and `Fake.csv` → Kaggle dataset format

---

## ⚙️ Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/Bharathnani123/fakenewsdetection.git
cd fakenewsdetection
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train & Evaluate

```bash
python fake-news.py
```

### Interactive Prediction

After training, you can type news content directly:

```
News> Aliens landed in New York today!
Prediction: ❌ FAKE
```

---

## 📈 Example Output

```
Accuracy: 92.30%
Confusion Matrix:
 [[620   8]
  [ 45 580]]

Classification Report:
              precision    recall  f1-score   support
FAKE            0.93       0.96       0.95       628
REAL            0.98       0.93       0.95       625
```

---

## 🚀 Future Improvements

* Add **deep learning models** (LSTM, BERT)
* Deploy as a **web app** (Streamlit / Flask)
* Provide a **REST API** for integration

---

