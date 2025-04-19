import openai
from dotenv import load_dotenv


# load variables from .env file
load_dotenv()

def generate_text(prompt):
    response = openai.chat.completions.create(
        model="davinci-002",
        messages=[
            {"role": "system", "content": "you are customer care representative for a car dealership."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0.5
    )
    return response.choices[0].message["content"].strip()

prompt = "I need help"
print(generate_text(prompt))