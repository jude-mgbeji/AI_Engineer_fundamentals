
from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Sample text
text = "Text vectorization is an essential step in NLP."

# Tokenize and convert to input IDs
inputs = tokenizer(text, return_tensors="pt")
print(inputs)

# Generate embeddings
outputs = model(**inputs)
embedding = outputs.last_hidden_state
print(embedding)

print(embedding.shape)  # Shape: [Batch size, Sequence length, Hidden size]