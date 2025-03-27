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
# print("ADD DUPLICATE DOCUMENT TO STORE")
# Add a duplicate document to the vector store. We do this to explain certain concepts later
# doc_to_add =  Document(page_content= '''Alright! So… How are the techniques used in data, business intelligence, or predictive analytics applied in real life? Certainly, with the help of computers. You can basically split the relevant tools into two categories—programming languages and software. Knowing a programming language enables you to devise programs that can execute specific operations. Moreover, you can reuse these programs whenever you need to execute the same action''', 
#                           metadata = {"course tittle": "Introduction toData and Data Science",
#                                       "lecture tittle": "Programming Languages & Software Employed in Data Science - All the Tools You Need"})
# vectorStore.add_documents(documents= [doc_to_add])

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

question = "What programming language do data scientist use?"

# NB: k indicates the number of documents to be returned
retrieved_documents = vectorStore.similarity_search(query= question, k= 5)

for i in retrieved_documents:
    lecture_tittle = i.metadata["lecture tittle"]
    print(f"Page Content: {i.page_content} \n ------------\nLecture tittle: {lecture_tittle} \n")

# Notice that there is a duplicate document returned due to the initial duplicate 
# we intentionally added. This duplicate have created redundancies in what is being retrived
# taking up the space for a more relevant document to be returned.
# This problem can be solved using the MAXIMAL MARGINAL RELEVANCE APPROACH