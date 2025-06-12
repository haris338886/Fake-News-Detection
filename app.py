import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('fake_or_real_news.csv')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [ps.stem(word) for word in tokens]
    return ' '.join(tokens)

df['cleaned_text'] = df['text'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text']).toarray()

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

joblib.dump(classifier, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'vectorizer_model.pkl')

model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        article = request.form['article']
        cleaned_article = preprocess_text(article)
        vectorized_article = vectorizer.transform([cleaned_article]).toarray()
        prediction = model.predict(vectorized_article)[0]
        if prediction == 1:
            result = "Real News"
        else:
            result = "Fake News"
        return render_template('index.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
