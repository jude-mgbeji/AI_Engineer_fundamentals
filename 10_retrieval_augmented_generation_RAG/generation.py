from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# load variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

embedding = OpenAIEmbeddings(model= "text-embedding-ada-002")

# In order to retrieve the embeddings from storage, we do the following
vectorStore = Chroma(persist_directory= r"10_retrieval_augmented_generation_RAG/vectorestore",
                                    embedding_function= embedding)

question = "What software do data scientist use?"

# NB: k indicates the number of documents to be returned
# the retriver object is an instance of runnable
retriever = vectorStore.as_retriever(search_type = 'mmr',
                                     search_kwargs = {'k': 3, 'lambda_mult': 0.7})

TEMPLATE = '''
Answer the following question:
{question}

To answer the question, use only the following context:
{context}

At the end of the respnse, specify the name of the lecture this context is taken from in the format:
Resources: "Lecture tittle"
where "Lecture tittle" should be substituted with the tittle of all resource lectures
'''

prompt_template = PromptTemplate.from_template(TEMPLATE)

# create the llm model component
chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=250)

output_parser = StrOutputParser()

chain = {'context': retriever, 'question': RunnablePassthrough()} | prompt_template | chat | output_parser

response = chain.invoke(question)

print(response)