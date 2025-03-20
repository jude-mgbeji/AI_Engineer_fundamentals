from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationSummaryMemory
from langchain_core.runnables import chain, RunnablePassthrough, RunnableLambda
from operator import itemgetter

# load variables from .env file
load_dotenv()

chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=250)

TEMPLATE = '''
You are Marv, a chatbot that relunctantly answers questions with sarcastice responses.

current conversation: 
{message_log}

Human: {question}
AI:
'''

chat_template = PromptTemplate.from_template(template=TEMPLATE)

chat_memory = ConversationSummaryMemory(llm= ChatOpenAI(), memory_key= "message_log")

@chain
def memory_chain(question):
    chain1 = (
    RunnablePassthrough.assign(
        message_log = RunnableLambda(chat_memory.load_memory_variables) | itemgetter("message_log"))
    | chat_template 
    | chat 
    | StrOutputParser()
    )

    response = chain1.invoke({'question': question})

    chat_memory.save_context(inputs= {'input': question}, outputs= {'output': response})

    print(chat_memory.load_memory_variables({}))

    return response

question = "Can you tell me an interesting fact I don't know about?"

response = memory_chain.invoke(question)
print(response)

response = memory_chain.invoke('can you ellaborate this fact')
print(response)