from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document 
from dotenv import load_dotenv
import os

# load variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# after splitting, notice that the doc content for page 4, 6 are from similar course tittle
# while that of page 20 is different 
# Convert this page content to vector embeddings

embedding = OpenAIEmbeddings(model= "text-embedding-ada-002")

# In order to retrieve the embeddings from storage, we do the following
vectorStore_from_directory = Chroma(persist_directory= r"10_retrieval_augmented_generation_RAG/vectorestore",
                                    embedding_function= embedding)

print(vectorStore_from_directory.get())
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
print("GET BY ID")
print(vectorStore_from_directory.get(ids= 'f8fcd417-d997-4cbb-be3c-582e50816359', include= ['embeddings']))

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
print("ADD NEW DOCUMENT")

# adding new document to the vector store
doc_to_add = Document(page_content= '''Alright! So… Let’s discuss the not-so-obvious differences between the terms analysis and analytics. Due to the similarity of the words, some people believe they share the same meaning, and thus use them interchangeably. Technically, this isn’t correct. There is, in fact, a distinct difference between the two. And the reason for one often being used instead of the other is the lack of a transparent understanding of both. So, let’s clear this up, shall we? First, we will start with analysis''', 
                          metadata = {"course tittle": "Introduction toData and Data Science",
                                      "lecture tittle": "nalysis vs Analytics"})

added_document_id = vectorStore_from_directory.add_documents([doc_to_add])

print(added_document_id)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
print(vectorStore_from_directory.get(added_document_id[0]))
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
print("UPDATE EXIXTING DOC")

doc_to_update =  Document(page_content= '''Great! We hope we gave you a good idea about the level of applicability of the most frequently used programming and software tools in the field of data science. Thank you for watching!''', 
                          metadata = {"course tittle": "Introduction toData and Data Science",
                                      "lecture tittle": "Programming Languages & Software Employed in Data Science - All the Tools You Need"})

vectorStore_from_directory.update_document(document_id=added_document_id[0], document= doc_to_update)

print(vectorStore_from_directory.get(added_document_id[0]))

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
print("DELETE EXIXTING DOC")

vectorStore_from_directory.delete(ids=added_document_id)

print(vectorStore_from_directory.get(added_document_id[0]))


