import nltk
# nltk.download('stopwords') //should be executed just once
from nltk.corpus import stopwords


# An important first step in data processing it to convert the data/text 
# into lower lower case to help maintain consistency.

text = "I am the best that I can be and my name is Jagaban"
lower_text = text.lower()
print(lower_text)

text_array = ["Go for it", 
              "You are the main guy in town",
                "Shame on you", "BADDEST GUY"]
lower_text_array = [x.lower() for x in text_array]
print(lower_text_array)

# NLTK Python package: The nltk (Natural Language Toolkit) module is one
#  of the most popular Python libraries for Natural Language Processing (NLP). 
# It provides tools for working with human language data, such as tokenisation, 
# stemming, lemmatisation, parsing, and more.

# REMOVING STOPWORDS : Remove words that dont necessarily add much meaning to the data
# we can do this using the nltk python library that has this words already labelled

en_stopwords = stopwords.words("english")
# print(en_stopwords)
sentence_stopwords = "It was too far to go the shop and he did not want her to walk"
sentence_no_stopwords_array = [word for word in sentence_stopwords.split()
                         if word not in en_stopwords]
# convert the array of no stopwords to a sentence
sentence_no_stopwords = ' '.join(sentence_no_stopwords_array)
print(sentence_no_stopwords)

# We can also either add or remove stopwords form the original data from the nltk library

# remove stopwords
en_stopwords.remove("go")

# Add stopwords
en_stopwords.append("not")

