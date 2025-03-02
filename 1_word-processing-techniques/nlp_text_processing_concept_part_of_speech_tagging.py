import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import spacy
import pandas as pd

# nltk.download('averaged_perceptron_tagger_eng')

# POS tagging is a foundational NLP technique that provides critical grammatical
#  context for text analysis. Libraries like nltk and spaCy make implementing POS 
# tagging straightforward, empowering applications like syntactic parsing, sentiment
#  analysis, and entity recognition.

# USING NLTK
text = "The quick brown fox jumps over the lazy dog."
# convert the text to a token
tokens = word_tokenize(text)
# perform POS tagging
token_tag = pos_tag(tokens)

print(token_tag)

# USING SPACY
# spaCy is another NLP library that offers faster and more efficient POS tagging.

# Load English tokenizer and POS tagger
nlp = spacy.load("en_core_web_sm")

# process the text
doc = nlp(text)

# Print tokens and their POS tags
for token in doc:
    print(token.text, token.pos_)

print("PANDAS Analysis >>>>>>>>>>>>>>")
# Using PANDAS to perform some analysis on the data
pos_df = pd.DataFrame(columns=["token", "pos_tag"])

# Add the data into a dataframe
for token in doc:
    pos_df = pd.concat([pos_df, pd.DataFrame.from_records([{"token": token.text, "pos_tag": token.pos_}])], ignore_index=True)

print(pos_df.head(15))