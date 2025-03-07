from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

text = "I love Hugging Face!"
input_ids = tokenizer(text, return_tensors="pt")
print(input_ids)
output = model(**input_ids)
output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

print(output)

logits = output.logits
predicted_class = torch.argmax(logits).item()
predicted_class_label = model.config.id2label[predicted_class]
print(predicted_class_label)
print(text)
