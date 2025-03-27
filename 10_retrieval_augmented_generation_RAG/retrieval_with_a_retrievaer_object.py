from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document 
from dotenv import load_dotenv
import os

# load variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

embedding = OpenAIEmbeddings(model= "text-embedding-ada-002")

# In order to retrieve the embeddings from storage, we do the following
vectorStore = Chroma(persist_directory= r"10_retrieval_augmented_generation_RAG/vectorestore",
                                    embedding_function= embedding)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

question = "What software do data scientist use?"

# NB: k indicates the number of documents to be returned
# the retriver object is an instance of runnable
retriever = vectorStore.as_retriever(search_type = 'mmr',
                                     search_kwargs = {'k': 3, 'lambda_mult': 0.7})

retrieved_documents = retriever.invoke(question)

for i in retrieved_documents:
    lecture_tittle = i.metadata["lecture tittle"]
    print(f"Page Content: {i.page_content} \n ------------\nLecture tittle: {lecture_tittle} \n")
