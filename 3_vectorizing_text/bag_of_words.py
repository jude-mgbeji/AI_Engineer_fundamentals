from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Bag of Words
	# •	Represents text as a collection (bag) of words, ignoring grammar and word order.
	# •	Each word in the vocabulary corresponds to a dimension in the vector space.
	# •	The vector value is typically the word count or presence (binary) in the text.

data = ['I love machine learning',
        'I love deep learning', 
        'I love natural language processing']

countVectorizer = CountVectorizer()

# Fit the data and transform it into a vector
vectorized_data = countVectorizer.fit_transform(data)

# Get the feature names
feature_names = countVectorizer.get_feature_names_out()
print(feature_names)

# Get the vectorized data
vectorized_data_array = vectorized_data.toarray()
print(vectorized_data_array)

# Get the vectorized data in a DataFrame
bag_of_words_df = pd.DataFrame(vectorized_data_array, columns=feature_names)

print(bag_of_words_df)

