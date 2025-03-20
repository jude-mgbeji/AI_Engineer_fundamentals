import os
import openai
from dotenv import load_dotenv

# load variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
print(openai_api_key)
client = openai.OpenAI()
completions = client.chat.completions.create(model="gpt-3.5-turbo",
                                             messages=[
                                                 {
                                                    "role": "system",
                                                    "content": "you witll be provided with a tweet, and your task is to classify its sentiment as positive, neutral or negative"
                                                },
                                                {
                                                    "role": "user",
                                                    "content": "This booke is extraorinary"
                                                },
                                                {
                                                    "role": "assistant",
                                                    "content": "POSITIVE"                                                   
                                                },
                                                {
                                                    "role": "user",
                                                    "content": "The movie was okay"
                                                },
                                                {
                                                    "role": "assistant",
                                                    "content": "NEUTRAL"                                                   
                                                },
                                                {
                                                    "role": "user",
                                                    "content": "I find the food to be very disgusting"
                                                },
                                                {
                                                    "role": "assistant",
                                                    "content": "NEGATIVE"                                                   
                                                },
                                                {
                                                    "role": "user",
                                                    "content": "I can read that book 1000 times over, it is so suspense filled each time"
                                                }                                                                                                
                                             ])
# evaluate the returned object
print(completions)

# retrive the task result
result = completions.choices[0].message.content
print(result)