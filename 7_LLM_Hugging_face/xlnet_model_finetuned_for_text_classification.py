from transformers import XLNetTokenizer, XLNetForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import re 
import datasets
import evaluate
# import random
from emoji import replace_emoji

# load the data
data_train = pd.read_csv(r'/Users/mgbejijude/Documents/ml-fundamental/7_LLM_Hugging_face/emotion-labels-train.csv')
data_test = pd.read_csv(r'/Users/mgbejijude/Documents/ml-fundamental/7_LLM_Hugging_face/emotion-labels-test.csv')
data_val = pd.read_csv(r'/Users/mgbejijude/Documents/ml-fundamental/7_LLM_Hugging_face/emotion-labels-val.csv')
print(data_train.head(5))
print(data_test.head(5))
print(data_val.head(5))

# concatenate the data to make cleaning and data preprocessing easier
data = pd.concat([data_train, data_test, data_val], ignore_index=True)
print(data.head(5))

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Preprocessing the data>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# clean the data
def clean_text(text):
    text = replace_emoji(text, replace="") # replace emojis
    text = re.sub(r'[^\w\s]', '', text) # remove punctuations
    return text

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Cleaning the data")
data['clean_text'] = data['text'].apply(clean_text)
print(data.head(10))

# Checking the data spread based on the labels
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Checking the data spread based on the labels")
print(data['label'].value_counts())
# balance the data
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Balancing the data")
data = data.groupby('label').apply(lambda x: x.sample(data['label'].value_counts().min()).reset_index(drop=True))
print(data['label'].value_counts())

# encode the labels
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Encoding the labels")
data["label_encoded"] = LabelEncoder().fit_transform(data["label"])
print(data.head(5))
NUM_CLASSES = len(data["label"].unique())
print(NUM_CLASSES)

# split the data
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Splitting the data")
train_split, test_split = train_test_split(data, test_size=0.2)
# further split the train data to get the validation data
train_split, val_split = train_test_split(train_split, test_size=0.1)
print(train_split.shape, test_split.shape, val_split.shape)

# create a new dataframe to store the split data with the clean text and label encoded

train_df = pd.DataFrame({
    "text": train_split["clean_text"],
    "label": train_split["label_encoded"]
})
test_df = pd.DataFrame({
    "text": test_split["clean_text"],
    "label": test_split["label_encoded"]
})
val_df = pd.DataFrame({
    "text": val_split["clean_text"],
    "label": val_split["label_encoded"]
})

# create a dataset object for the train, test and val data
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Creating the dataset object")
train_dataset = datasets.Dataset.from_dict(train_df)
test_dataset = datasets.Dataset.from_dict(test_df)
val_dataset = datasets.Dataset.from_dict(val_df)
print(train_dataset)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Creating the dataset dict")
dataset_dict = datasets.DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
})
print(dataset_dict)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Creating the data embeddings>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# load the tokenizer
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Loading the tokenizer")
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

# tokenize the data
def tokenize_function(data):
    return tokenizer(data["text"], padding=True, max_lenght=128, truncation=True)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Tokenizing the data")
tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
print(tokenized_datasets)

# reduce the size of the dataset for faster testing of the model
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Reducing the size of the dataset")
smaller_tokenized_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
smaller_tokenized_test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))
print(smaller_tokenized_train_dataset)
print(smaller_tokenized_test_dataset)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Fine tuning the model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# load the model
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Loading the model")
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased',
                                                        num_labels=NUM_CLASSES,
                                                        id2label={0: "anger", 1: "fear", 2: "joy", 3:"sadness"})

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# define the training arguments
training_args = TrainingArguments(output_dir="./xlnet_test_trainer", eval_strategy="epoch", num_train_epochs=3)

data_collector = DataCollatorWithPadding(tokenizer=tokenizer)

# create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=smaller_tokenized_train_dataset,
    eval_dataset=smaller_tokenized_test_dataset,
    data_collator=data_collector,
    compute_metrics=compute_metrics
)

# train the model
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Training the model")
trainer.train()

trainer.evaluate()

# save the model
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Saving the model")
model.save_pretrained("./trained_xlnet_model/xlnet_model_finetuned_for_text_classification")
tokenizer.save_pretrained("./trained_xlnet_model/xlnet_tokenizer_finetuned_for_text_classification")














