from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load GPT-2 (successor to GPT)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenize and generate text
inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=50)
print(tokenizer.decode(outputs[0]))