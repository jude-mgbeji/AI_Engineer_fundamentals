from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool

# load variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

embedding = OpenAIEmbeddings(model= "text-embedding-ada-002")

# In order to retrieve the embeddings from storage, we do the following
vectorStore = Chroma(persist_directory= r"./10_retrieval_augmented_generation_RAG/vectorestore",
                                    embedding_function= embedding)

retriever = vectorStore.as_retriever(search_type= 'mmr', 
                                     search_kwargs = {'k': 3, 'lambda_mult': 0.7})

retriever_tool = create_retriever_tool(retriever= retriever, 
                                       name= "Intoduction to Data and Data Science Course Lectures", 
                                       description= '''For any questions regarding the 
                                       Intoduction to Data and Data Science Course Lectures,
                                       you must use this tool.''')

print(f"tool name: {retriever_tool.name} \n description: {retriever_tool.description} \n arguments: {retriever_tool.args}")

response = retriever_tool.invoke("What are the programming languages used by a data scientist?")

print(response)
