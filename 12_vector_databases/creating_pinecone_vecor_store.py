from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override= True)

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# connect to pinecone
pc = Pinecone(api_key= pinecone_api_key)

print(pc.list_indexes())

index_name = "my-index"
dimension = 3
metric = "cosine"

# creating the vector store or index if it doesexist
if index_name in [index.name for index in pc.list_indexes()]:
    # pc.delete_index(index_name)
    print(f"{index_name} succcessfully already exist")
else:
    pc.create_index(
        name= index_name,
        dimension= dimension,
        metric= metric,
        spec= ServerlessSpec(cloud= 'aws', region= 'us-east-1')
    )
    print(f"{index_name} does not exist, now created")


print(pc.list_indexes())

