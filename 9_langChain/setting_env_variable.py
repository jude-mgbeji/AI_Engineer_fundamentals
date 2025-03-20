import os
from dotenv import load_dotenv

# load variables from .env file
load_dotenv()

for key, value in os.environ.items():
    if key == "OPENAI_API_KEY":
        print(f'{key}: {value}')    

print(os.getenv("OPENAI_API_KEY"))