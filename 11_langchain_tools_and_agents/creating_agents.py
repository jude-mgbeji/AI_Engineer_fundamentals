from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool
from langchain import hub
from platform import python_version
from langchain.agents import create_tool_calling_agent, AgentExecutor


# load variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=250)

wikipedis_api = WikipediaAPIWrapper()

wikipedis_tool = WikipediaQueryRun(api_wrapper= wikipedis_api)

embedding = OpenAIEmbeddings(model= "text-embedding-ada-002")

# In order to retrieve the embeddings from storage, we do the following
vectorStore = Chroma(persist_directory= r"./10_retrieval_augmented_generation_RAG/vectorestore",
                                    embedding_function= embedding)

retriever = vectorStore.as_retriever(search_type= 'mmr', 
                                     search_kwargs = {'k': 3, 'lambda_mult': 0.7})

retriever_tool = create_retriever_tool(retriever= retriever, 
                                       name= "Intoduction-to-Data-and-Data-Science-Course-Lectures", 
                                       description= '''For any questions regarding the 
                                       Intoduction to Data and Data Science Course Lectures,
                                       you must use this tool.''')

@tool
def get_python_version() -> str:
    '''Useful for questions regarding the version of python currently used.'''
    return python_version()

# create a list of the tools
tools = [wikipedis_tool, retriever_tool, get_python_version]

# This pulls an already created prompt template from the langchain hub
prompt_template = hub.pull("hwchase17/openai-tools-agent")
print(prompt_template.pretty_print())
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n\n")

# create the agent
agent = create_tool_calling_agent(llm= chat, 
                                  tools= tools, 
                                  prompt= prompt_template)

# create the AgentExecutor to carry out the agent decisions
agent_executor = AgentExecutor(agent= agent,
                               tools= tools,
                               verbose= True,
                               return_intermediate_steps= True)

response = agent_executor.invoke({"input": "Could you tell me the version of python I am currently using?"})
print(response)

response = agent_executor.invoke({"input": '''Could you list the programmming languages a 
                                  data scientist should know? 
                                  Additionally could you tell me who their creators are?'''})
print(response)

# NB: when we invoke the agent directly without the agentExecutor, we get an AgentAction object retuned 
# this action is then passed as part of the intermediate_steps tuple which is an argument when 
# calling the agent.invoke()
# when there is no further action to be carried out by the agent, it returns an AgentFinish
# object which contains the output