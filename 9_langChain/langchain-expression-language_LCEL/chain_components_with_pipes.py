from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# load variables from .env file
load_dotenv()

output_instruction = CommaSeparatedListOutputParser().get_format_instructions()

# create a prompt template component. Notice that we did not use the HumanMessage object 
# or a template, instead we create a template directly using a tuple
prompt_template = ChatPromptTemplate.from_messages([("human", "I've recently adopted a {pet}, can you suggest three {pet} names? \n" + output_instruction )])

# create the llm model component
chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=250)

# create the output parser component
output_parser = CommaSeparatedListOutputParser()

# using langchain expression language, we can create a chain of components using a pipe, 
# such the output of one becomes the input of the next in order
chain = prompt_template | chat | output_parser

output = chain.invoke({"pet": "dog"})

print(output)