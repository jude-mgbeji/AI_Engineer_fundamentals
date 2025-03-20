from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory, CombinedMemory
from langchain.globals import set_verbose  

# the langchain verbose function is simply for debugging purpose, to ellaborate logs etc
set_verbose(True)


# load variables from .env file
load_dotenv()

chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=250)

# We want to combine two memory objects in our chatbot. Also note that we have decided to use the
# String prompt template approach just to show diversity

TEMPLATE = '''
You are Marv, a chatbot that relunctantly answers questions with sarcastice responses.

past messages: 
{message_buffer_log}

conversation summary: 
{message_summary_log}

Human: {question}
AI:
'''

chat_template = PromptTemplate.from_template(template=TEMPLATE)

# In defining the memory object, Notice we did not pass the chat_history attribute, this is because
# there is no existing or prior chat history. Also we set return_messages to false because we want 
# the messages as strings
bufferMemory = ConversationBufferMemory(memory_key= 'message_buffer_log', input_key= 'question', return_messages=False)
summaryMemory = ConversationSummaryMemory(llm= ChatOpenAI(), memory_key= 'message_summary_log', input_key='question', return_messages=False)
combinedMemory = CombinedMemory(memories=[bufferMemory, summaryMemory])

print(combinedMemory.load_memory_variables({}))

chain = LLMChain(llm=chat, prompt=chat_template, memory= combinedMemory)

response = chain.invoke({'question': "Can you give an interesting fact I probably didn't know about?"})

print(response['text'])

response = chain.invoke({'question': "Can you ellaborate on this fact?"})

print(response['text'])