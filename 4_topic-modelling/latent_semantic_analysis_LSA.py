import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import gensim
import gensim.corpora as corpora
from gensim.models import LsiModel

# How LSA Works
	# 1.	Create a Document-Term Matrix (DTM):
    	# •	Text is vectorized into numerical form, typically using a Bag of Words (BoW) or TF-IDF representation.
	# 2.	Apply Singular Value Decomposition (SVD):
	    # •	Decompose the DTM into matrices  U ,  S , and  V^T .
	# 3.	Reduce Dimensions:
	    # •	Select the top  k  singular values (topics) in  S , reducing noise while retaining meaningful patterns.
	# 4.	Interpret Topics:
	    # •	Examine the top words in each topic (columns of  V^T ) to assign human-readable labels.

data = pd.read_csv(r'/Users/mgbejijude/Documents/ml-fundamental/4_topic-modelling/news_articles.csv')
print(data.head())
print(data.info())
print(data['content'][0])

# A FUNTION FOR DATA CLEANING
def clean_text(text, en_stopwords, stemmer):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join([word for word in text.split() if word not in en_stopwords])
    tokenized_text = word_tokenize(text)
    stemmed_text = [stemmer.stem(word) for word in tokenized_text]
    return stemmed_text

# CLEANING THE DATA
print("<<<<<<<<<<<<<<<<<<<<<<<<DATA CLEANING>>>>>>>>>>>>>>>>>>>>>>>>")
en_stopwords = stopwords.words('english')
ps = PorterStemmer()
data['cleaned_content'] = data['content'].apply(lambda x: clean_text(x, en_stopwords, ps))
print(data['cleaned_content'][0])

# CREATING A DICTIONARY OF UNIQUE TOKENS WORDS IN THE DATA
print("<<<<<<<<<<<<<<<<<<<<<<<<CREATING A DICTIONARY>>>>>>>>>>>>>>>>>>>>>>>>")
dictionary = corpora.Dictionary(data['cleaned_content'])

# VECTORISING THE DATA USING THE BASIC BAG OF WORDS MODEL
# NB: This is a more simpler way of implementing the bag of words model compared to the one in the previous script
print("<<<<<<<<<<<<<<<<<<<<<<<<VECTORISING THE DATA WITH BAG OF WORDS MODEL>>>>>>>>>>>>>>>>>>>>>>>>")
doc_term_matrix = [dictionary.doc2bow(doc) for doc in data['cleaned_content']]

lsa_model = LsiModel(doc_term_matrix, num_topics=2, id2word=dictionary)
print(lsa_model.print_topics(num_topics=2, num_words=5))

# The output of the LSA model is similar to that of the LDA model. The main difference is that the LSA model uses
#  Singular Value Decomposition (SVD) to reduce the dimensionality of the document-term matrix, while the LDA model 
# uses a probabilistic approach to model the topics in the data.
# The LSA model is useful for reducing noise in the data and identifying meaningful patterns in the text. It can be 
# used to extract topics from text data and provide insights into the underlying structure of the data.
# In this example, we trained an LSA model on a dataset of news articles and extracted two topics from the data. 
# The model identified the top words associated with each topic and provided a human-readable label for each topic.
#  This information can be used to analyze the content of the news articles and gain insights into the topics covered in the data.
# Overall, the LSA model is a powerful tool for analyzing text data and extracting meaningful patterns from large datasets.
#  It can be used to identify topics, trends, and patterns in text data and provide valuable insights for data analysis and 
# decision-making.
