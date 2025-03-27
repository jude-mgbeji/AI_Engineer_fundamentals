from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example texts
text1 = "I love programming in Python."
text2 = "Python is my favorite programming language."
text3 = "I had fufu for lunch."

# Generate embeddings
embedding1 = model.encode(text1)
embedding2 = model.encode(text2)
embedding3 = model.encode(text3)

print(f"Vector magnitudes are {np.linalg.norm(embedding1)} , {np.linalg.norm(embedding2)} , {np.linalg.norm(embedding3)}")

# Compute cosine similarity
cosine_similarity = util.cos_sim(embedding1, embedding2)
cosine_similarity2 = util.cos_sim(embedding1, embedding3)
cosine_similarity3 = util.cos_sim(embedding2, embedding3)

print(f"Cosine Similarity: {cosine_similarity.item():.4f}")
print(f"Cosine Similarity: {cosine_similarity2.item():.4f}")
print(f"Cosine Similarity: {cosine_similarity3.item():.4f}")