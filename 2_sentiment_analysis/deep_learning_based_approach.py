import transformers
from transformers import pipeline

# Deep Learning Approach:
	# •	Uses more advanced models like Recurrent Neural Networks (RNN), 
    # Long Short-Term Memory networks (LSTM), and Transformer-based models 
    # (e.g., BERT) to analyze the context and detect sentiment in more complex text.

# •	These models can capture the relationships between words and phrases in a sentence,
#   allowing them to understand nuances, sarcasm, and context better than simpler methods.

# •	However, they require more data and computational resources to train and may be
#   more complex to implement and interpret than rule-based or lexicon-based approaches.

# •	Deep learning models can be fine-tuned on specific datasets or tasks to improve
#   performance, making them versatile for various sentiment analysis applications.

# USING PRE-TRAINED TRANSFORMERS

sentiment_pipeline = pipeline("sentiment-analysis")

# It should be noted that no models was supplied to the pipeline function, 
# so the default model was used.
# The default model is distilbert-base-uncased-finetuned-sst-2-english and
#  revison 714eb0f from the huggingface model hub.
# Using a pipeline without a model name and revision is not recommended in production code.
# It is better to specify the model name and revision to ensure reproducibility and compatibility.

sentence_1 = "I had a great time at the movie it was really funny"
sentence_2 = "I had a great time at the movie but the parking was terrible"
sentence_3 = "I had a great time at the movie but the parking wasn't great"
sentence_4 = "I want to see a movie"

print(sentence_1)
print(sentiment_pipeline(sentence_1))

print(sentence_2)
print(sentiment_pipeline(sentence_2))

print(sentence_3)
print(sentiment_pipeline(sentence_3))

print(sentence_4)
print(sentiment_pipeline(sentence_4))
