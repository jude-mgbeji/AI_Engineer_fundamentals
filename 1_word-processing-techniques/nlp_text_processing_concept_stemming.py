import nltk
from nltk.stem import PorterStemmer

ps = PorterStemmer()

# Stemming is a simple and effective way to reduce words to their base form,
#  making it an essential preprocessing step in NLP tasks like text 
# classification, sentiment analysis, and search engines. However, it should
#  be used with caution to avoid over- or under-stemming errors.

connect_tokens =["connected", "connects", "connecting", "connectivity", "connect"]
learn_token =["learner", "learning", "learners", "learn", "learns", "learned"]

for token in connect_tokens:
    print(token, " : ", ps.stem(token))

for token in learn_token:
    print(token, " : ", ps.stem(token))