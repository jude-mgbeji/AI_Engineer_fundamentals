import pandas as pd 
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy



# Read the data from the csv
bbc_news_df = pd.read_csv(r"/Users/mgbejijude/Documents/ml-fundamental/1_word-processing-techniques/bbc_news.csv")
print(bbc_news_df.columns)
print(bbc_news_df["title"][0])
tittle_df = pd.DataFrame(bbc_news_df["title"])
print(tittle_df.head)
# DATA CLEANING
print("<<<<<<<<<<<<<<<<<<DATA CLEANING>>>>>>>>>>>>>>>>>>>>")
print("Lower case >>>>>>>>>>>>>>>>>>>>>>>>>>>")
tittle_df["tittle_lowercase"] = tittle_df["title"].str.lower()
print(tittle_df["tittle_lowercase"][0])


print("Removing stop words >>>>>>>>>>>>>>>>>>>>>>>>>>>")
en_stopwords = stopwords.words("english")
tittle_df["no_stopwords"] = tittle_df["tittle_lowercase"].apply(lambda x: " ".join(
    [word for word in x.split() 
     if word not in en_stopwords] 
))
print(tittle_df["no_stopwords"][0])

print("Removing punctuations >>>>>>>>>>>>>>>>>>>>>>>>>>>")
tittle_df["no_stopwords_no_punc"] = tittle_df["no_stopwords"].apply(lambda x: 
                                                                    re.sub(r"[^\w\s]", "", x))
print(tittle_df["no_stopwords_no_punc"][0])

print("Tokenize >>>>>>>>>>>>>>>>>>>>>>>>>>>")
tittle_df["clean_token"] = tittle_df["no_stopwords_no_punc"].apply(lambda x: word_tokenize(x))
print(tittle_df["clean_token"][0])

print("Lemmatizing >>>>>>>>>>>>>>>>>>>>>>>>>>>")
lemmatizer = WordNetLemmatizer()
tittle_df["lemmatized_token"] = tittle_df["clean_token"].apply(lambda x: [
    lemmatizer.lemmatize(token) for token in x
])
print(tittle_df["lemmatized_token"][0])

print("convert or unpack the tokens into a single list >>>>>>>>>>>>>>>>>>>>>>>>>>>")
clean_token_array = sum(tittle_df["clean_token"], [])
print(clean_token_array)

print("<<<<<<<<<<<<<<<<<<<<POS TAGGING>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
nlp = spacy.load("en_core_web_sm")
doc = nlp(" ".join(clean_token_array))
pos_tag_df = pd.DataFrame(columns=["token", "pos_tag"])
for token in doc:
    pos_tag_df = pd.concat([pos_tag_df, pd.DataFrame.from_records([{"token": token.text, "pos_tag": token.pos_}])], ignore_index=True)
print(pos_tag_df.head)

print("token frequency count >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
token_frequency_count_df = pos_tag_df.groupby(["token", "pos_tag"]).size().reset_index(name="count").sort_values(by="count", ascending=False)
print(token_frequency_count_df.head)

print("most common nouns >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
nouns = token_frequency_count_df[token_frequency_count_df.pos_tag == "NOUN"][0:10]
print(nouns)

print("most common verbs >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
verbs = token_frequency_count_df[token_frequency_count_df.pos_tag == "VERB"][0:10]
print(verbs)

print("<<<<<<<<<<<<<<<<<<<<NAMED ENTITY RECOGNITION>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
named_entity_df = pd.DataFrame(columns=["token", "label"])
for token in doc.ents:
    named_entity_df = pd.concat([named_entity_df, pd.DataFrame.from_records([{"token": token.text, "label": token.label_}])], ignore_index=True)
print(named_entity_df.head)

print("token frequency count >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
ner_token_frequency_count_df = named_entity_df.groupby(["token", "label"]).size().reset_index(name="count").sort_values(by="count", ascending=False)
print(ner_token_frequency_count_df.head)

print("top 10 most common person >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
persons = ner_token_frequency_count_df[ner_token_frequency_count_df.label == "PERSON"][0:10]
print(persons)






