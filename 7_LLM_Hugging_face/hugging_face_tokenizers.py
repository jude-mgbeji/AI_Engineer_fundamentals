from transformers import AutoTokenizer

# The Hugging Face Tokenizers library provides efficient tokenization tools for working with large language models.
# It supports a wide range of models, including BERT, GPT-2, RoBERTa, XLNet, and many others.
# The library provides tokenization functions to convert text into token IDs that can be fed into the models.
# It also supports decoding token IDs back into human-readable text and converting tokens to token IDs.

print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Using the BERT model")
model = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model)
text = "Hello, world!"
input_ids = tokenizer(text)
print(input_ids)
# Output: [101, 7592, 1010, 2088, 999, 102]
# The tokenizer converts the input text into a sequence of token IDs that the model can understand 
# including the special tokens.

decoded_text = tokenizer.decode(input_ids["input_ids"])
print(decoded_text)
# Output: [CLS] hello, world! [SEP]
# The tokenizer can also decode the token IDs back into human-readable text.
# notice the special tokens [CLS] and [SEP] at the beginning and end of the sequence.

tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['hello', ',', 'world', '!']
# The tokenizer also provides a list of tokens corresponding to the input text.

token_id = tokenizer.convert_tokens_to_ids(tokens)
print(token_id)
# Output: [7592, 1010, 2088, 999]
# You can also convert tokens back to token IDs using the tokenizer.

print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Using the XLNET model")
model2 = "xlnet-base-cased"
tokenizer2 = AutoTokenizer.from_pretrained(model2)
text2 = "Hello, world!"
input_ids2 = tokenizer2(text2)
print(input_ids2)
# Output: [17, 47, 315, 4, 3]
# The tokenizer converts the input text into a sequence of token IDs that the model can understand
# including the special tokens.

decoded_text2 = tokenizer2.decode(input_ids2["input_ids"])
print(decoded_text2)
# Output: hello, world! <sep> <cls>
# The tokenizer can also decode the token IDs back into human-readable text.
# notice the special tokens <sep> and <cls> at the end of the sequence.

tokens2 = tokenizer2.tokenize(text2)
print(tokens2)
# Output: ['▁Hello', ',', '▁world', '!']
# The tokenizer also provides a list of tokens corresponding to the input text.

token_id2 = tokenizer2.convert_tokens_to_ids(tokens2)
print(token_id2)
# Output: [17, 47, 315, 4]
# You can also convert tokens back to token IDs using the tokenizer.

# NB:
# Here we have used the AutoTokenizer class to automatically load the appropriate tokenizer for the model.
# We can also use specific tokenizer classes like BertTokenizer, GPT2Tokenizer, RobertaTokenizer, etc.,
# depending on the model we are working with.




