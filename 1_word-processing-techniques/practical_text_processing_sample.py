import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import re

data = pd.read_csv(r"/Users/mgbejijude/Documents/ml-fundamental/1_word-processing-techniques/tripadvisor_hotel_reviews.csv")
print(data.info())
# take a peek at the data
print(data.head())
# take a look at the first review in the data 
print(data["Review"][0])
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<START DATA PROCESSING>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# convert the data to lowercase
print(">>>>>>>>>>>>>>>convert text data to lowecase")
data["Review_lowercase"] = data["Review"].str.lower()
print(data.head())
# remove stopwords from text data
print(">>>>>>>>>>>>>>>>removing stopwords")
en_stopwords = stopwords.words("english")
# rmove "not" from stopwords list
en_stopwords.remove("not")
data["Review_no_stopwords"] = data["Review_lowercase"].apply(lambda x: 
                               " ".join(
                                    [word for word in x.split()
                                     if word not in en_stopwords]
                                     )
                                )
print(data["Review_no_stopwords"][0])
# removing punctuations
print(">>>>>>>>>>>>>>>>>Removing punctuations")
# first substitute * with star in the reviews
data["Review_no_stopwords_no_puncs"] = data['Review_no_stopwords'].apply(lambda x:
                                                                         re.sub(r"\*", "star", x))
# remove any string that is not a word or whitespace
data["Review_no_stopwords_no_puncs"] = data['Review_no_stopwords_no_puncs'].apply(lambda x:
                                                                                  re.sub(r"[^\w\s]", "", x))
print(data.head())
# tokenizing
print(">>>>>>>>>>>>>>>>>>>>>Tokenize")
data["tokenize"] = data["Review_no_stopwords_no_puncs"].apply(lambda x:
                                                              word_tokenize(x))
print(data["tokenize"][0])
# Stemming
print(">>>>>>>>>>>>>>>>>>>>>>>Stemming")
stemmer = PorterStemmer()
data["stemmed"] = data["tokenize"].apply(lambda tokens:
                                         [stemmer.stem(token) for token in tokens])
print(data.head())
# Comparing a stemmed token to a lemmatized token
print(">>>>>>>>>>>>>>>>>>>>>>>Comparing stemming to lemmatization")
lemmatizer = WordNetLemmatizer()
data["lemmatized"] = data["tokenize"].apply(lambda tokens:
                                         [lemmatizer.lemmatize(token) for token in tokens])
print(data.head())
print("<<<<<<<<<<<<<<<<<<<<WE NOW HAVE A CLEAN DATASET>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# To do some analysis on the clean data using N-grams and Pandas
print(">>>>>>>>>>>>>>>>>>>>Analizing the clean data using n-gram and pandas")
# Flatten the list of list in the lemmatized column
clean_tokens = sum(data["lemmatized"], [])
unigram = nltk.ngrams(clean_tokens, 1)
bigram = nltk.ngrams(clean_tokens, 2)
# using pandas to get word frequency from unigram and bigram
print(">>>>>>>>>>>>>>> unigram word frequency")
print(pd.Series(unigram).value_counts())
print(">>>>>>>>>>>>>>> bigram word frequency")
print(pd.Series(bigram).value_counts())






