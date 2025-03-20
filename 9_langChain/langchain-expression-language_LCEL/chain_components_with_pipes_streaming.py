from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# load variables from .env file
load_dotenv()

# create a prompt template component. Notice that we did not use the HumanMessage object 
# or a template, instead we create a template directly using a tuple
prompt_template = ChatPromptTemplate.from_messages([("human", "I've recently adopted a {pet}, which is a {breed}, can you suggest several training tips?" )])

# create the llm model component
chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=250)

# using langchain expression language, we can create a chain of components using a pipe, 
# such the output of one becomes the input of the next in order
chain = prompt_template | chat

# When we use the stream function on a chain, usually, we want to stream the output  
# tokens as it is being generated 
output = chain.stream({"pet": "dragon", "breed": "night fury"})

for i in output:
    print(i.content, end="")