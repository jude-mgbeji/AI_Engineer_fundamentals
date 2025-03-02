from transformers import AutoTokenizer, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
import torch
import tensorflow as tf

print("<<<<<<<<<<<<<<<<<<<<<<<<<<< Using pytorch")
# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenize input
text = "I love Hugging Face!"
input_ids_pt = tokenizer(text, return_tensors="pt")
print(input_ids_pt)
# Output: {'input_ids': tensor([[  101,  1045,  2293,  6279,  1998, 12090,   102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])}

# Perform inference
output = model(input_ids = input_ids_pt["input_ids"], attention_mask = input_ids_pt["attention_mask"])
print(output)
# # Output: SequenceClassifierOutput(loss=None, logits=tensor([[-2.7276,  2.8056]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)

with torch.no_grad():
    logits = output.logits

# Get predicted class
predicted_class = torch.argmax(logits).item()
print(predicted_class)
# Output: 1
predicted_class_label = model.config.id2label[predicted_class]
print(predicted_class_label)
# Output: 'LABEL_1'

print("<<<<<<<<<<<<<<<<<<<<<<<<<<< Using tensorflow")

# Load pre-trained model and tokenizer
model2 = TFAutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenize input
input_ids_tf = tokenizer(text, return_tensors="tf")
print(input_ids_tf)

# Perform inference
output2 = model2(input_ids = input_ids_tf["input_ids"], attention_mask = input_ids_tf["attention_mask"])
print(output2)
# Output: TFSequenceClassifierOutput(loss=None, logits=<tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[-2.7276,  2.8056]], dtype=float32)>, hidden_states=None, attentions=None)

logits2 = output2.logits

# Get predicted class
predicted_class2 = tf.argmax(logits2).numpy()[0].item()
print(predicted_class2)
# Output: 1
predicted_class_label2 = model.config.id2label[predicted_class2]
print(predicted_class_label2)
print(text)


