from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

#  TF-IDF (Term Frequency-Inverse Document Frequency)
	# •	Improves on BoW by scaling word importance:
	# •	Term Frequency (TF): How often a word appears in a document. It is calcuted as: 
        # Number of times a word appears in a document divided by Total number of words in the document
	# •	Inverse Document Frequency (IDF): Reduces the weight of common words (e.g., “is”, “the”). It is calculated as:
        # log(Total number of documents divided by Number of documents with the word in it)

data = ['I love machine learning',
        'I love deep learning', 
        'I love natural language processing']

tfidfVectorizer = TfidfVectorizer()

# Fit the data and transform it into a vector
vectorized_data = tfidfVectorizer.fit_transform(data)

# Get the feature names
feature_names = tfidfVectorizer.get_feature_names_out()

# Get the vectorized data
vectorized_data_array = vectorized_data.toarray()

# Get the vectorized data in a DataFrame
tf_idf_df = pd.DataFrame(vectorized_data_array, columns=feature_names)
print(tf_idf_df)