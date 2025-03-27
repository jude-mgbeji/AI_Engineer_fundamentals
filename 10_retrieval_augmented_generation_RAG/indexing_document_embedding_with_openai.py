from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
import numpy as np
from dotenv import load_dotenv
import os

# load variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

docx_loader = Docx2txtLoader(r"10_retrieval_augmented_generation_RAG/docs/Introduction_to_Data_and_Data_Science 2.docx")
docx_pages = docx_loader.load()

# This splitter splits the document based on headers in the doc, Identified by #, ##, ### etc.
# the number of # depicts the header level in the document and should be factored
#  into the tuple argument. 
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "course tittle"), ("##", "lecture tittle")])

splitted_docx_content = splitter.split_text(docx_pages[0].page_content)

print(len(splitted_docx_content))

# go through each page in the doc and remove new line characters
for content in splitted_docx_content:
    content.page_content = " ".join(content.page_content.split())

char_splitter = CharacterTextSplitter(separator=".", chunk_size=500, chunk_overlap=50)

char_splitted_docx_pages = char_splitter.split_documents(splitted_docx_content)

print(len(char_splitted_docx_pages))
print((char_splitted_docx_pages[3]))
print((char_splitted_docx_pages[5]))
print((char_splitted_docx_pages[19]))

# after splitting, notice that the doc content for page 4, 6 are from similar course tittle
# while that of page 20 is different 
# Convert this page content to vector embeddings

embedding = OpenAIEmbeddings(model= "text-embedding-ada-002")

vector1 = embedding.embed_query(char_splitted_docx_pages[3].page_content)
vector2 = embedding.embed_query(char_splitted_docx_pages[5].page_content)
vector3 = embedding.embed_query(char_splitted_docx_pages[19].page_content)

print(len(vector1), len(vector2), len(vector3))

# In calculating the magnitude of this vectors, we see that the vectors is normalized because the result is 1

print(np.linalg.norm(vector1), np.linalg.norm(vector2), np.linalg.norm(vector3))

# Note that, how similar this text represensented by vectors are is usually 
# a factor of how far apart the vectors are from each other in the verctor space
# There are several ways of getting the distance between two vectors in space. 
# One of which is using the dot products of normalized vectors
# if the vectors are not normalized, then we use the cosine of similarities approach
# The higher the dot product of two vectors, the closer they are. Which translate to similarity

print(np.dot(vector1, vector2))
print(np.dot(vector1, vector3))
print(np.dot(vector2, vector3))

# vector 1 and 2 are closer hence, they have similar content

