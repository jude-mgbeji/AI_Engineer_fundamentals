from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Sentiment Analysis is a natural language processing (NLP) task that involves determining
# the emotional tone or attitude expressed in a piece of text. It is typically used to 
# classify text into positive, negative, or neutral sentiment categories, but more
# granular classifications (e.g., happy, sad, angry, etc.) can also be performed.

# Lexicon-Based or Rule-based Approach:
	# •	Relies on predefined lists of words associated with positive or negative sentiment
        # (e.g., “happy,” “good,” “bad,” “sad”).
	# •	Sentiment score is calculated based on the presence and frequency of these words
        # in the text.

# 	•	Both VADER and TextBlob are lexicon-based because they rely on predefined word 
#       lists (or lexicons) to classify sentiment.
#   •	They calculate sentiment scores based on the frequency and context of words in 
#       a sentence, often adding rules to improve accuracy, like handling negations or 
#       intensifiers (e.g., “very happy” would increase the positivity).

# 	•	While these tools are easy to use and can provide a quick sentiment analysis, they
#       may not be as accurate as more sophisticated machine learning models, especially
#       for complex or nuanced text.

sentence_1 = "I had a great time at the movie it was really funny"
sentence_2 = "I had a great time at the movie but the parking was terrible"
sentence_3 = "I had a great time at the movie but the parking wasn't great"
sentence_4 = "I want to see a movie"




# USING TEXTBLOB
# Easy to use but often does not understand sarcasm or context
print("<<<<<<<<<<<<<<<<<<<<<USING TEXTBLOB >>>>>>>>>>>>>>>>>>>>>>>>>>>>")

sentiment_score_1 = TextBlob(sentence_1)
print(sentence_1)
print(sentiment_score_1.sentiment.polarity)

sentiment_score_2 = TextBlob(sentence_2)
print(sentence_2)
print(sentiment_score_2.sentiment.polarity)

sentiment_score_3 = TextBlob(sentence_3)
print(sentence_3)
print(sentiment_score_3.sentiment.polarity)

sentiment_score_4 = TextBlob(sentence_4)
print(sentence_4)
print(sentiment_score_4.sentiment.polarity)


# USING VADER
# More sophisticated than TextBlob, but still has limitations. Specifically 
# tuned for social media text. It gives a score for positive, negative, and neutral
# sentiments, along with a compound score.
print("<<<<<<<<<<<<<<<<<<<<<USING VADER >>>>>>>>>>>>>>>>>>>>>>>>>>>>")

vader_sentiment = SentimentIntensityAnalyzer()

print(sentence_1)
print(vader_sentiment.polarity_scores(sentence_1))

print(sentence_2)
print(vader_sentiment.polarity_scores(sentence_2))

print(sentence_3)
print(vader_sentiment.polarity_scores(sentence_3))

print(sentence_4)
print(vader_sentiment.polarity_scores(sentence_4))



