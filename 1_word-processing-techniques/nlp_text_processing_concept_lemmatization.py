import nltk
from nltk.stem import WordNetLemmatizer

# nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()


# Lemmatization is the process of reducing a word to its base or dictionary 
# form, known as a lemma, while ensuring that the resulting word is 
# meaningful and grammatically correct. Unlike stemming, which can produce
#  non-words
# Lemmatization is a critical step in text preprocessing, especially for 
# tasks that require semantic understanding and accuracy. While it’s slower 
# than stemming, it provides valid base forms of words, 
# making it preferable for high-precision NLP tasks.

connect_tokens =["connected", "connects", "connecting", "connectivity", "connect"]
learn_token =["learner", "learning", "learners", "learn", "learns", "learned"]

for token in connect_tokens:
    print(token, " : ", lemmatizer.lemmatize(token))

for token in learn_token:
    print(token, " : ", lemmatizer.lemmatize(token))