from nltk import ngrams
from nltk.tokenize import word_tokenize
import pandas as pd
import matplotlib.pyplot as plt

# An N-Gram is a continuous sequence of N items (tokens) from a given text. 
# These items can be words, characters, or even phonemes. N-Grams are 
# commonly used in Natural Language Processing (NLP) for tasks like language
#  modelling, text generation, and predictive text.
# N-Grams are fundamental tools in NLP that provide insights into local word
#  patterns and are the backbone of many language models and applications.
#  However, they are often complemented by more advanced techniques, 
# such as transformer-based models, to overcome their limitations in handling 
# long-range dependencies and large vocabularies.

string = r"N-Grams are fundamental tools in NLP that provide insights into local word patterns and are the backbone of many language models and applications. However, they are often complemented by more advanced techniques, such as transformer-based models, to overcome their limitations in handling long-range dependencies and large vocabularies."
tokenized_string = word_tokenize(string)
print("TOKENIZED TEXT : ", tokenized_string)
unigram = ngrams(tokenized_string, 1)
bigram = ngrams(tokenized_string, 2)
trigram = ngrams(tokenized_string, 3)

print("UNIGRAM  : ", unigram)

# Using Pandas to perform analysis
print("<<<<<<<<<<<<<<<<<<<<PANDAS SERIES ANALYSIS UNIGRAM>>>>>>>>>>>>>>>>>>>>")
unigram_analysis = pd.Series(unigram).value_counts()
print(unigram_analysis[:10])

print("<<<<<<<<<<<<<<<<<<<<PANDAS SERIES ANALYSIS BIGRAM>>>>>>>>>>>>>>>>>>>>")
bigram_analysis = pd.Series(bigram).value_counts()
print(bigram_analysis[:10])

print("<<<<<<<<<<<<<<<<<<< PLOTING IN A GRAPH >>>>>>>>>>>>>>>>>>>>>>")
unigram_analysis[:10].sort_values().plot.barh(color="green", width=.9, figsize =(12,8))
plt.title("10 Most frequently occurring unigram")



