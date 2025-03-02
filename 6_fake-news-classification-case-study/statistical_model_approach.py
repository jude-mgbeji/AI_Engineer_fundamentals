import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = pd.read_csv(r'/Users/mgbejijude/Documents/ml-fundamental/6_fake-news-classification-case-study/fake_news_data.csv')
print(data.head())

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# text preprocessing
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)

data['cleaned_text'] = data['text'].apply(preprocess_text)
print(data['cleaned_text'][0])

# vectorizing the data
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['cleaned_text']).toarray()
print(tfidf_matrix)

labels = data['fake_or_factual'].apply(lambda x: 1 if x == 'Fake News' else 0)
print(labels)

# splitting the data into trained and test data
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=42)

# training the model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

# making predictions
y_predictions = logistic_regression.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_predictions))
print("Classification Report:")
print(classification_report(y_test, y_predictions))

# saving the model and using it to make predictions
joblib.dump(logistic_regression, 'fake_news_classifier.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

loaded_model = joblib.load('fake_news_classifier.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

text = "Macbook is an microsoft product."
text = preprocess_text(text)
text_vectorized = loaded_vectorizer.transform([text]).toarray()
prediction = loaded_model.predict(text_vectorized)
print("Prediction:", prediction)