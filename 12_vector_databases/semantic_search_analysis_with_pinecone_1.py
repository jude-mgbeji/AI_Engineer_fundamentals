import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv, find_dotenv

data = pd.read_csv(r'12_vector_databases/data/course_descriptions.csv', encoding='cp1252')
print(data.info())
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")

def create_course_description(row):
    return f'''The course name is {row['course_name']}, the slug is {row['course_slug']}, 
    the technology is {row['course_technology']} and the course topic is {row['course_topic']}'''

data['course_description_new'] = data.apply(lambda row: create_course_description(row), axis= 1)
print(data.head(5))
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")

load_dotenv(find_dotenv(), override= True)

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# connect to pinecone
pc = Pinecone(api_key= pinecone_api_key)

# define the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

index_name = "365-courses-store"

# create a vector store that corresponds to the dimension of the model chosen for embedding
if index_name in [index.name for index in pc.list_indexes()]:
    print(f"{index_name} succcessfully already exist\n\n")
else:
    pc.create_index(
        name= index_name,
        dimension= model.get_sentence_embedding_dimension(),
        metric= "cosine",
        spec= ServerlessSpec(cloud= 'aws', region= 'us-east-1')
    )
    print(f"{index_name} does not exist, now created\n\n")

# get the vector store
vector_store = pc.Index(name= index_name)

def vectorize_row(row):
    text = ','.join([str(row['course_description']), str(row['course_description_new']), str(row['course_description_short'])])
    embedding = model.encode(text, show_progress_bar=False)
    return embedding

data['embedding'] = data.apply(vectorize_row, axis = 1)

vector_to_upsert = []
for index, row in data.iterrows():
    row_tuple = (str(row['course_name']), row["embedding"].tolist())
    vector_to_upsert.append(row_tuple)

vector_store.upsert(vector_to_upsert)

print("Data upserted to pinecone vector store")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")

# we want to search for courses related to clustering
query = 'clustering' 
query_embedding = model.encode(query, show_progress_bar= False).tolist()

result = vector_store.query(
    vector= [query_embedding],
    top_k= 3,
    include_values= True
)

print(result)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")

for match in result['matches']:
    print(f"Matched Item ID: {match['id']}, score: {match['score']}")

'''
It should be noted that the result for this use case does not give accurate
 result because of how the original data is structured i.e information about the courses 
 are contained in the course section which we dont have in the provided data
'''