from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.llm import LLMChain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.memory import ConversationSummaryMemory
from langchain.globals import set_verbose  

# the langchain verbose function is simply for debugging purpose, to ellaborate logs etc
set_verbose(True)


# load variables from .env file
load_dotenv()

chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=250)

system_message = SystemMessage(content='''You are Marv, a chatbot that relunctantly answers questions with sarcastice responses.''')
human_message_template = HumanMessagePromptTemplate.from_template(template='{question}')

# create a placeholder for intermediate messages logged
messagePlaceholder = MessagesPlaceholder(variable_name= 'message_log')

# create the prompt template object. NOTE the order in the argument passed
chat_template = ChatPromptTemplate.from_messages([system_message, messagePlaceholder, human_message_template])

# initialize the InMemoryChatMessageHistory object
chat_history = InMemoryChatMessageHistory()
chat_history.add_user_message('Hi')
chat_history.add_ai_message("you really do know how to make an entrance don't you?")

# create the summary memory object. when the return_messages is true, 
# then messages are returned as a list. returned as a string if false
# the difference this and the ConversationBufferMemory is that it stores only a summary of 
# AI-HUMAN interaction instead of the entire conversation
summaryMemory = ConversationSummaryMemory(llm= ChatOpenAI(), memory_key="message_log", 
                                        chat_memory= chat_history,
                                        return_messages=True)
print(summaryMemory.load_memory_variables({}))

chain = LLMChain(prompt= chat_template, llm= chat, memory= summaryMemory)

response = chain.invoke({'question': "Can you give an interesting fact I probably didn't know about?"})

print(response['text'])

response = chain.invoke({'question': "Can you ellaborate on this fact?"})

print(response['text'])

response = chain.invoke({'question': "Can you tell me another fact?"})

print(response['text'])

# Notice how only two pairs of Human-AI interaction is retained for each level of interactions
