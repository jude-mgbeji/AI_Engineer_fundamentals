import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel

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

# TRAINING THE LDA MODEL
print("<<<<<<<<<<<<<<<<<<<<<<<<TRAINING THE LDA MODEL>>>>>>>>>>>>>>>>>>>>>>>>")
lda_model = LdaModel(doc_term_matrix, num_topics=2, id2word=dictionary, passes=50)
print(lda_model.print_topics(num_topics=2, num_words=5))

