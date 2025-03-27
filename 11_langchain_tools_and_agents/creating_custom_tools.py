from dotenv import load_dotenv
import os
from langchain_core.tools import tool
from platform import python_version

# load variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

@tool
def get_python_version() -> str:
    '''Useful for questions regarding the version of python currently used.'''
    return python_version()

print(f"tool name: {get_python_version.name} \n description: {get_python_version.description} \n arguments: {get_python_version.args}")

response = get_python_version.invoke({})

print(response)
