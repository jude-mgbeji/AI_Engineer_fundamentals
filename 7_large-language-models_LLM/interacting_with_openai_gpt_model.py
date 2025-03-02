import openai
from config import openai_api_key

openai.api_key = openai_api_key

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