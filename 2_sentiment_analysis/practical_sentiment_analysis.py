import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from transformers import pipeline

data = pd.read_csv(r"/Users/mgbejijude/Documents/ml-fundamental/2_sentiment_analysis/book_reviews_sample.csv")
print(data.columns)
print(data["reviewText"][0])

print("<<<<<<<<<<<<<<<<<<DATA CLEANING [REMOVING PUNCTUATIONS AND LOWER CASING]>>>>>>>>>>>>>>>>>>>>")
data["review_text_clean"] = data["reviewText"].apply(lambda x: re.sub(r"[^\w\s]", "", x.lower()))
print(data["review_text_clean"][0])

print("<<<<<<<<<<<<<<<<<<SENTIMENT ANALYSIS WITH VADER>>>>>>>>>>>>>>>>>>>>")
vader_sentiment = SentimentIntensityAnalyzer()

data["vader_sentiment_score"] = data["review_text_clean"].apply(lambda x: vader_sentiment.polarity_scores(x).get("compound"))
print(data["vader_sentiment_score"][0])
bins = [-1, -0.1, 0.1, 1]
labels = ["negative", "neutral", "positive"]
data["vader_sentiment"] = pd.cut(data["vader_sentiment_score"], bins=bins, labels=labels)
print(data["vader_sentiment"][0])

# Visualizing the data
sentiment_counts = data["vader_sentiment"].value_counts()

plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['red', 'gray', 'green'])
plt.title('Sentiment Analysis of Book Reviews with vader')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.show()

print("<<<<<<<<<<<<<<<<<<SENTIMENT ANALYSIS WITH TRANSFORMERS>>>>>>>>>>>>>>>>>>>>")

sentiment_pipeline = pipeline("sentiment-analysis")
data["transformers_sentiment"] = data["review_text_clean"].apply(lambda x: sentiment_pipeline(x)[0].get("label"))
print(data["transformers_sentiment"][0])

# Visualizing the data
sentiment_counts = data["transformers_sentiment"].value_counts()

plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['red', 'gray', 'green'])
plt.title('Sentiment Analysis of Book Reviews with transformers')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.show()

