from transformers import pipeline

# The Hugging Face Transformers library is one of the most popular Python libraries 
# for working with state-of-the-art Natural Language Processing (NLP) models. 
# It provides pre-trained models and tools to easily integrate them into various tasks 
# like text classification, translation, summarization, and more.

# The library supports a wide range of models, including BERT, GPT-2, RoBERTa, T5, and many others.
# It also provides interfaces for fine-tuning models on custom datasets, loading models and saving them for later use,

# Using the default pre-trained model for sentiment analysis
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Sentiment Analysis using default model")
sentiment_classifier = pipeline("sentiment-analysis")
sendtiment_result = sentiment_classifier("I love the Transformers library!")
print(sendtiment_result)

# performaing named entity recognition with a specific model
print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Named Entity Recognition using specific model")
ner_classifier = pipeline("ner", model="dslim/bert-base-NER")
ner_result = ner_classifier("Hugging Face is a great company founded in Paris.")
print(ner_result)

# NB: 
# Zero-shot classification is a natural language processing (NLP) technique where a model classifies
#  data into categories it has never seen during training. This eliminates the need for labeled training
#  data for each specific task or domain. Instead, the model uses general knowledge acquired during 
# pretraining.

# perfroming zero-shot classification with a specific model
print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Zero-shot classification")
zeroShot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
text = "Who is the author of Harry Potter?"
candidate_labels = ["science", "sports", "politics", "history"]
zeroShot_result = zeroShot_classifier(text, candidate_labels)
print(zeroShot_result)
