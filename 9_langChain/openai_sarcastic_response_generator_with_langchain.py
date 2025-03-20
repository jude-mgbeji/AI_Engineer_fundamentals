from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# load variables from .env file
load_dotenv()

chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=250)
response = chat.invoke(''' I've recently adopted a dog, could you suggest some dog names?''')

print(response)
print(response.content)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> Using the system and user role based messages")

chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=250)

message_system = SystemMessage(content='''You are Marv, a chatbot that relunctantly answers questions with sarcastice responses.''')
message_human = HumanMessage(content='''I've recently adopted a dog, could you suggest some dog names?''')

response = chat.invoke([message_system, message_human])

print(response.content)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> Using the AImessages to teach the model how to response")

# NB: It should be noted that this technique is not an ideal case, as this will only increase
#  cost as the number of input tokens increases. It is ideal to use system and human messages

chat = ChatOpenAI(model_name = 'gpt-4',
                  seed= 365,
                  temperature= 0,
                  max_completion_tokens=250)

message_dog = HumanMessage(content='''I've recently adopted a dog, could you suggest some dog names?''')
message_ai_dog = AIMessage(content='''Oh, absolutely. Because I'm sure your dog is just waiting to be 
                           named by a sarcastic AI. How about "Bark Twain" if it's a literary hound,''')
message_cat = HumanMessage(content='''I've recently adopted a cat, could you suggest some cat names?''')
message_ai_cat = AIMessage(content='''Oh, sure! How about "Whiskers" or "Meowster"? Those are pawsitively
                            perfect for your new feline friend.''')

message_fish = HumanMessage(content='''I've recently adopted a fish, could you suggest some fish names?''')


response = chat.invoke([message_dog, message_ai_dog, message_cat, message_ai_cat, message_fish])

print(response.content)
