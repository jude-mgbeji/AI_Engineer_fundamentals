from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun


# load variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=250)

TEMPLATE = '''
Turn the following user input into a wikipedia search query. Don't answer the question:

{input}
'''

# create the WikipediaAPIWrapper object, this helps to fetch articles through the load() method
# also to fetch page summaries through the run() method
wikipedis_api = WikipediaAPIWrapper()

# create the wikipedia tool using the WikipediaQueryRun class. The WikipediaQueryRun class is an 
# instance of the runnable class. This implies that it has the invoke() function.
# NB: Tools generally have a defined name and description
wikipedis_tool = WikipediaQueryRun(api_wrapper= wikipedis_api)

print(f"tool name: {wikipedis_tool.name} \n description: {wikipedis_tool.description} \n arguments: {wikipedis_tool.args}")

response = wikipedis_tool.invoke("python")
print(response)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n\n")

# create the prompt template
prompt_template = PromptTemplate.from_template(template= TEMPLATE)

chain = prompt_template | chat | StrOutputParser() | wikipedis_tool

response = chain.invoke({"input": "Who is the creator of the python programming language?"})

print(response)

