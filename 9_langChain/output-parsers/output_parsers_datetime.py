from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from custom_output_parser import  DatetimeOutputParser


# load variables from .env file
load_dotenv()

chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=250)

# here in creating the message object we use a string formatter where we pass the 
# string message and the output format instructions
human_message = HumanMessage(content= f'''When did Nigeria gain Independence?
                 {DatetimeOutputParser().get_format_instructions()}            
''')

print(human_message.content)

response = chat.invoke([human_message])

output_parser = DatetimeOutputParser()

output = output_parser.invoke(response)

print(output)
