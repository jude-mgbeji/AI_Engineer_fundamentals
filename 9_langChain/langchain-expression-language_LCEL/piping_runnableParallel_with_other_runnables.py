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

time_prompt_template = ChatPromptTemplate.from_template('''
                                                        I'm an intermediate level programmer.
                                                        Consider the following literature:
                                                        {books}
                                                        Also, consider the following projects:
                                                        {projects}
                                                        Roughly how much time would it take  me to complete the literate and the projects?
                                                        ''')

# create the llm model component
chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=500)

ouput_parser = StrOutputParser()

book_chain = books_prompt_template | chat | ouput_parser
project_chain = project_prompt_template | chat | ouput_parser

# in order to run both chains in parallel, we use RunnableParallel class 
# which takes in a dictionary of the the chains to run
parallel_runnable_chain = RunnableParallel({'books': book_chain, 'projects': project_chain})

# Because the result of the parallel_runnable_chain is the input to the time chain, 
# we pipe both as a new single chain
time_chain = parallel_runnable_chain | time_prompt_template | chat | ouput_parser

# the above line can also be written as:
time_chain2 = {'books': book_chain, 'projects': project_chain} | time_prompt_template | chat | ouput_parser


response = time_chain.invoke({"programming_language": "python"})
print(response)

# this will visually show the relationships between all component runnables within the chain
print(time_chain.get_graph().print_ascii())

