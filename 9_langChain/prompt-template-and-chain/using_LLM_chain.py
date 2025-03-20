from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import HumanMessagePromptTemplate, AIMessagePromptTemplate,FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain.chains.llm import LLMChain


# load variables from .env file
load_dotenv()

chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=250)

TEMPLATE_HUMAN = '''I've recently adopted a {pet}, Could you suggest some {pet} names?'''
TEMPLATE_AI = '{response}'

message_template_human = HumanMessagePromptTemplate.from_template(TEMPLATE_HUMAN)
message_template_ai = AIMessagePromptTemplate.from_template(TEMPLATE_AI)

example_template = ChatPromptTemplate.from_messages([message_template_human, message_template_ai])

# create a list of dictiomarries that defines examples of the template input variables
examples = [
    {'pet': 'dog', 
     'response': '''Oh, absolutely. Because I'm the perfect entity to ask about dog names, 
     being an AI and all. How about "Bark Twain" or "Sir Wag-a-lot"? Or maybe "Bitey McChewface"? Those are top-notch, right?'''},
     {'pet': 'cat',
      'response': '''Oh, sure! How about "Whiskers" or "Meowster"? Those are pawsitively
                            perfect for your new feline friend.'''}
]

few_shot_prompt = FewShotChatMessagePromptTemplate(examples= examples, example_prompt= example_template)

chat_template = ChatPromptTemplate.from_messages([few_shot_prompt, message_template_human])

# Although this is an obsollette way of using the chain instead of the more intune LCEL approach,
#  it is important we just know it to understand chatbot memory concepts etc
chain = LLMChain(llm= chat, prompt= chat_template)

response = chain.invoke({'pet': 'fish'})

print(response)
