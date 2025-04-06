from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv, find_dotenv
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

'''
In this script, we will be loading test data from the hugging face fineweb datasets.
Which is simply a collection of cleaned web data from random crawling.
We will also be using the huggingface sentence transformer model for embedding/vectorizing
'''

load_dotenv(find_dotenv(), override= True)

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# load the dataset from huggingface
data = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
print(data.features)

# define the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# connect to pinecone
pc = Pinecone(api_key= pinecone_api_key)

index_name = "text-vector-store"

# create a vector store that corresponds to the dimension of the model chosen for embedding
if index_name in [index.name for index in pc.list_indexes()]:
    print(f"{index_name} succcessfully already exist")
else:
    pc.create_index(
        name= index_name,
        dimension= model.get_sentence_embedding_dimension(),
        metric= "cosine",
        spec= ServerlessSpec(cloud= 'aws', region= 'us-east-1')
    )
    print(f"{index_name} does not exist, now created")

# get the vector store
vector_store = pc.Index(name= index_name)

# for convenience let us embed just 10000 records
limit = 10000

# prepare and vectorize the data
vectors_to_upsert =[]

for i, item in enumerate(data):
    if i >= limit:
        break

    text = item['text']
    id = item['id']
    language = item['language']

    # embed the text using the model
    embedding = model.encode(text, show_progress_bar= False).tolist()

    metadata = {'language': language}

    item_tuple = (id, embedding, metadata)

    # append as a tuple to the list
    vectors_to_upsert.append(item_tuple)

# load/upsert vectorized data to pinecone in batches
batch_size = 1000

for i in range(0, len(vectors_to_upsert), batch_size):
    batch = vectors_to_upsert[i : i + batch_size]
    vector_store.upsert(vectors= batch)

'''It should be noted that the pinecone upsert() function takes in array of tuples as aegument.
In constructing the tuple, the first item in the tuple is saved as the ID in the vector store.
The second item is the embedding and the third item is a metadata (which is usually a dictionary)
'''