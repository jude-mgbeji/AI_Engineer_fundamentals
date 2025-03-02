import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# nltk.download('punkt_tab') # //should be executed just once


# In Natural Language Processing (NLP), 
# a token is a single unit of text that is
#  meaningful for analysis.
# Tokenisation is the process of splitting
#  text into smaller units (tokens) to make it easier for a machine to 
# process and analyse. It’s a crucial preprocessing step in NLP tasks, 
# as most models and algorithms work with tokens rather than raw text.

# sentence tokenization
string = r"He cat's name is Luna, Her dog's name is Max"
sentence_token = sent_tokenize(string)
print(sentence_token)

# Word tokenization
word_token = word_tokenize(string)
print(word_token)

# Note that the first character after tokenization is done is capitalised,
# this is why it is important to first convert to lower case before processing


