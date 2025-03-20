from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain_core.chat_history import InMemoryChatMessageHistory

# load variables from .env file
load_dotenv()

chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=250)

# initialize the InMemoryChatMessageHistory object
chat_history = InMemoryChatMessageHistory()

# add some human and AI messages to the history manually
chat_history.add_user_message("Can you give an interesting fact I probably didn't know about?")
chat_history.add_ai_message("Sure, did you know that the longest place name on the planet is 85 letters long?")
print(chat_history)

# converts history to a list of messages
chat_history_list = chat_history.messages

human_message_template = HumanMessagePromptTemplate.from_template(template='{follow-up-question}')

prompt_template = ChatPromptTemplate.from_messages(chat_history_list + [human_message_template])

chain = LLMChain(prompt=prompt_template, llm=chat)

response = chain.invoke({'follow-up-question': 'what is the name?'})

print(response)

response = chain.invoke({'follow-up-question': 'what is the meaning?'})

print(response)

# NB: Notice how the chatbot does not remember that it already gave the name of the place from the question