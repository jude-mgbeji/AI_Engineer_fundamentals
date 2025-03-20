from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# load variables from .env file
load_dotenv()

books_prompt_template = ChatPromptTemplate.from_template('''What are the five most important {programming_language} books for intermediatiate level?
                                 Answer only by listing the tools.''')

project_prompt_template = ChatPromptTemplate.from_template('''What are the five {programming_language} projects for intermediatiate level?
                                 Answer only by listing the tools.''')

# create the llm model component
chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=250)

ouput_parser = StrOutputParser()

book_chain = books_prompt_template | chat | ouput_parser
project_chain = project_prompt_template | chat | ouput_parser

# in order to run both chains in parallel, we use RunnableParallel class 
# which takes in a dictionary of the the chains to run
parallel_runnable_chain = RunnableParallel({'books': book_chain, 'projects': project_chain})

response = parallel_runnable_chain.invoke({"programming_language": "python"})
print(response)

# this will visually show the relationships between all component runnables within the chain
print(parallel_runnable_chain.get_graph().print_ascii())

