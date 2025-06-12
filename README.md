# 📰 Fake News Detection Using Machine Learning

## 📌 Overview

This project aims to detect **fake news articles** using **Natural Language Processing (NLP)** and **machine learning**. The objective is to classify news content as *Real* or *Fake* based solely on the textual information. A **Multinomial Naive Bayes** model was trained on TF-IDF-transformed data, achieving an accuracy of around **90%**. A simple **Flask web app** allows users to input news text and get real-time predictions.

---

## 🔍 Dataset

- **Source**: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- **Description**:
  - Contains news articles labeled as **REAL** or **FAKE**
  - Fields include:
    - `title`: headline of the article
    - `text`: full article content
    - `label`: REAL or FAKE

---

## 💻 Tools & Libraries

- **Language**: Python  
- **Libraries Used**:
  - `Pandas`, `NumPy` – Data manipulation
  - `NLTK` – Text preprocessing (stopwords, punctuation removal, tokenization)
  - `Scikit-learn` – TF-IDF vectorization, Naive Bayes model, evaluation
  - `Flask` – Web app for live fake news detection

---

## 📊 Project Stages

1. **Data Preprocessing**:
   - Removed punctuation, stopwords
   - Converted text to lowercase
   - Tokenized text using `NLTK`

2. **Feature Extraction**:
   - Transformed text data into numerical features using **TF-IDF Vectorizer**
   - Used 5000 top terms for vectorization

3. **Model Training**:
   - Trained a **Multinomial Naive Bayes** model on the TF-IDF vectors
   - Split the dataset into training and test sets

4. **Model Evaluation**:
   - Achieved ~90% accuracy
   - Evaluated using **confusion matrix**, **precision**, **recall**, and **F1-score**

5. **Web Deployment**:
   - Built a **Flask application**
   - Users can paste any news article to get a prediction instantly

---
