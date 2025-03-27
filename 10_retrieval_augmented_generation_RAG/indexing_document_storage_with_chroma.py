from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
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

# after splitting, notice that the doc content for page 4, 6 are from similar course tittle
# while that of page 20 is different 
# Convert this page content to vector embeddings

embedding = OpenAIEmbeddings(model= "text-embedding-ada-002")

# create the vector storage. This will create the embeddings for each page 
# in the document list and store it in the defined persist directory
vectorStore = Chroma.from_documents(documents=char_splitted_docx_pages, 
                                    embedding= embedding, 
                                    persist_directory= r"10_retrieval_augmented_generation_RAG/vectorestore")



