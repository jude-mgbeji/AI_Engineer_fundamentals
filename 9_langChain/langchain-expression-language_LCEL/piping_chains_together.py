from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# load variables from .env file
load_dotenv()

# NB: the RunnablePassthrough class is a Runnable that is  used to return exactly 
# what is passed into it. example
print(RunnablePassthrough().invoke([1,2,3]))

job_tools_prompt_template = ChatPromptTemplate.from_template('''What are the five most important tools a {job_tittle} needs?
                                 Answer only by listing the tools.''')

learning_startegy_prompt_template = ChatPromptTemplate.from_template('''Considering the tools provided,
                                                                     develop a strategy for effectively learning and mastering them:
                                                                     {tools}''')

# create the llm model component
chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=2500)

ouput_parser = StrOutputParser()

job_tools_chain = job_tools_prompt_template | chat | ouput_parser | {"tools": RunnablePassthrough()}
learning_strategy_chain = learning_startegy_prompt_template | chat | ouput_parser

combined_chain = job_tools_chain | learning_strategy_chain

response = combined_chain.invoke({"job_tittle": "data engineer"})

print(response)