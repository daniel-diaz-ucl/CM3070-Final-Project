from transformers import BertForSequenceClassification, BertTokenizer

# Specify the model name
model_name = 'bert-base-uncased'

# Download the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Save the tokenizer and model locally
tokenizer.save_pretrained('./bert-base-uncased')
model.save_pretrained('./bert-base-uncased')
