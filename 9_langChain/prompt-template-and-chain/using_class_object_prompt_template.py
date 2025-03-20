from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate

# load variables from .env file
load_dotenv()

chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=250)

TEMPLATE_SYSTEM = '{description}'
TEMPLATE_HUMAN = '''I've recently adopted a {pet}, Could you suggest some {pet} names?'''

message_template_system = SystemMessagePromptTemplate.from_template(template=TEMPLATE_SYSTEM)
message_template_human = HumanMessagePromptTemplate.from_template(template=TEMPLATE_HUMAN)

chat_template = ChatPromptTemplate.from_messages([message_template_system, message_template_human])

chat_value = chat_template.invoke({'description': '''The chatbot should relunctantly answer questions
                                    with sarcastic responses.''',
                                    'pet': 'dog'})

response = chat.invoke(chat_value)

print(response.content)