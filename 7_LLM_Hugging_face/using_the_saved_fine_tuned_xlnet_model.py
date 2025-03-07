from transformers import XLNetForSequenceClassification, XLNetTokenizer, pipeline

# Using the saved model

fine_tuned_model = XLNetForSequenceClassification.from_pretrained("./trained_xlnet_model/xlnet_model_finetuned_for_text_classification")
tokenizer =  XLNetTokenizer.from_pretrained("./trained_xlnet_model/xlnet_tokenizer_finetuned_for_text_classification")

clf = pipeline("text-classification", fine_tuned_model, tokenizer=tokenizer)

text = "I love ML, it makes me so excited each time i train a model"

answer = clf(text, top_k =None)

print(answer)

