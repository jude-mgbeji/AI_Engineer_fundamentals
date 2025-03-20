from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

# load variables from .env file
load_dotenv()

chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=250)

human_message = HumanMessage(content="Can you give an interesting fact I probably didn't know about?")

response = chat.invoke([human_message])

string_output_parser = StrOutputParser()

string_output = string_output_parser.invoke(response)

print(string_output)