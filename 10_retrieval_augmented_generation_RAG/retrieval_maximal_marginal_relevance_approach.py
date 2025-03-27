from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document 
from dotenv import load_dotenv
import os

'''
Maximal Marginal Relevance (MMR) is a technique used to rank or select items (e.g., documents, sentences, or search results) by balancing two objectives:
	1.	Relevance to a query.
	2.	Diversity to reduce redundancy.

How Does MMR Work?

MMR operates by iteratively selecting items that maximize the following objective:


MMR}(D_i) = lamda * {Sim}(D_i, Q) - (1 - \lambda) * {Sim}(D_i, D_j)


Where:
	•	 D_i : Candidate document or item being evaluated.
	•	 Q : Query or target.
	•	 S : Set of already selected items.
	•	 {Sim}(D_i, Q) : Similarity between the candidate  D_i  and the query  Q  (relevance).
	•	 {Sim}(D_i, D_j) : Similarity between the candidate  D_i  and already selected item  D_j  (redundancy).
	•	 \lambda : A tunable parameter ( 0 \leq \lambda \leq 1 ) controlling the trade-off between relevance and diversity.

Key Points:
	•	If  \lambda  is high ( \approx 1 ), MMR focuses more on relevance.
	•	If  \lambda  is low ( \approx 0 ), MMR prioritizes diversity by penalizing redundancy more heavily.
'''

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
retrieved_documents = vectorStore.max_marginal_relevance_search(query= question, 
                                                                k= 3,
                                                                lambda_mult=0.7)

for i in retrieved_documents:
    lecture_tittle = i.metadata["lecture tittle"]
    print(f"Page Content: {i.page_content} \n ------------\nLecture tittle: {lecture_tittle} \n")
