📰 Fake News Detection Using Machine Learning
📌 Overview
This project focuses on detecting fake news articles using Natural Language Processing (NLP) and machine learning techniques. The goal is to classify news content as Real or Fake based solely on text. A Naive Bayes model was trained on TF-IDF-transformed text data, achieving around 90% accuracy. A Flask web application was also developed, allowing users to paste news text and instantly receive a prediction.

🔍 Dataset
Source: Kaggle - Fake and Real News Dataset

Description: The dataset contains labeled news articles — real news collected from verified sources and fake news collected from unreliable or satirical websites. It includes fields such as title, text, and label (REAL/FAKE).

💻 Tools & Libraries
Python

Pandas, NumPy

NLTK (Natural Language Toolkit)

Scikit-learn (TF-IDF, Naive Bayes, model evaluation)

Flask (for web deployment)

📊 Project Stages
Data Cleaning & Preprocessing (removal of stopwords, punctuation, etc.)

TF-IDF Vectorization of text

Model Training using Multinomial Naive Bayes

Model Evaluation (accuracy, confusion matrix)

Flask Web App Deployment
