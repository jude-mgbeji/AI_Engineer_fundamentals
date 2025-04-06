import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv, find_dotenv

data = pd.read_csv(r'12_vector_databases/data/course_section_descriptions.csv', encoding='cp1252')
print(data.info())
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")
data['unique_id'] = data['course_id'].astype(str) + '-' + data['section_id'].astype(str)
data['metadata'] = data.apply(lambda row: {
    'course_name': row['course_name'],
    'section_name': row['section_name'],
    'section_description': row['section_description'],
}, axis= 1)

print(data.head(5))
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")

load_dotenv(find_dotenv(), override= True)

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# connect to pinecone
pc = Pinecone(api_key= pinecone_api_key)

# define the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def vectorize_row(row):
    text = f'''{row['course_name']} {row['course_technology']} {row['course_description']} {row['section_description']}'''
    return model.encode(text, show_progress_bar= False)

data['embedding'] = data.apply(vectorize_row, axis = 1)

index_name = "365-courses-section-store"

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

# array of tuple
vector_to_upsert = [(row["unique_id"], row['embedding'].tolist(), row['metadata']) for index, row in data.iterrows()]

vector_store.upsert(vectors= vector_to_upsert)

print("Data upserted to pinecone vector store")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")

# we want to search for courses related to clustering
query = 'clustering' 
query_embedding = model.encode(query, show_progress_bar= False).tolist()

result = vector_store.query(
    vector= [query_embedding],
    top_k= 12,
    include_metadata = True
)
score_threshold = 0.3

for match in result['matches']:
    if match['score'] >= score_threshold:
        course_details = match.get('metadata', {})
        course_name = course_details.get('course_name', 'N/A')
        section_name = course_details.get('section_name', 'N/A')
        section_description = course_details.get('section_description', 'N/A')

        print(f"Matched Item ID: {match['id']}, Score: {match['score']}")
        print(f"Course: {course_name} \nSection: {section_name} \nDescription: {section_description}")