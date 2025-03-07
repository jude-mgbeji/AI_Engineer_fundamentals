from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Tokenize input
inputs = tokenizer("This is an example sentence.", return_tensors="pt")
outputs = model(**inputs)

print(outputs)
# Output: SequenceClassifierOutput(loss=None, logits=tensor([[-0.1233,  0.4567]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)

logits = outputs.logits
predicted_class = torch.argmax(logits).item()
print(predicted_class)