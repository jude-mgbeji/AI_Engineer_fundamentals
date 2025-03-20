import os
import openai
from dotenv import load_dotenv

# load variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

completions = client.chat.completions.create(model="gpt-3.5-turbo",
                                             messages=[
                                                 {
                                                    "role": "system",
                                                    "content": "You are Marv, a chatbot that relunctantly answers questions with sarcastice responses."
                                                 },
                                                 {
                                                     "role": "user",
                                                     "content": "I recently adopted a cat, can you suggest some cat names?"
                                                 }
                                             ])
# retrive the task result
result = completions.choices[0].message.content
print(result)

# Adding more openAI API parameters 
print(">>>>>>>>>>>>>>>>>>>>>>>>>>> setting more openAI API parameters ")

completions = client.chat.completions.create(model="gpt-3.5-turbo",
                                             messages=[
                                                 {
                                                     "role": "user",
                                                     "content": "Can you explain what a black hole is?"
                                                 }
                                             ],
                                             max_tokens=250, # restricts the model output to the set number of tokens
                                             temperature=0, #this controls the level of randomness of the generated output. it ranges from 0-2. The higher the value, the higher the randomness. Usuaally the default temperature is 1.
                                             seed=365
                                             )
# retrive the task result
result = completions.choices[0].message.content
print(result)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>> streaming the API response ")

completions = client.chat.completions.create(model="gpt-3.5-turbo",
                                             messages=[
                                                 {
                                                     "role": "user",
                                                     "content": "Can you explain what a black hole is?"
                                                 }
                                             ],
                                             max_tokens=250, # restricts the model output to the set number of tokens
                                             temperature=0, #this controls the level of randomness of the generated output. it ranges from 0-2. The higher the value, the higher the randomness. Usuaally the default temperature is 1.
                                             seed=365, stream=True
                                             )
# retrive the task result
for i in completions:
    print(i.choices[0].delta.content, end="")



