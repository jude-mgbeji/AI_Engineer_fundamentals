from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# load variables from .env file
load_dotenv()

chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=250)

print(CommaSeparatedListOutputParser().get_format_instructions())

# here in creating the message object we use a string formatter where we pass the 
# string message and the output format instructions
human_message = HumanMessage(content= f'''I've recently adopted a dog, could you suggest some dog names?
                 {CommaSeparatedListOutputParser().get_format_instructions()}            
''')

response = chat.invoke([human_message])

output_parser = CommaSeparatedListOutputParser()

output = output_parser.invoke(response)

print(output)