from transformers import (DistilBertForSequenceClassification, DistilBertTokenizer,
                          AutoModelForSequenceClassification, AutoTokenizer)

# Specify the model names
distilbert_model_name = 'distilbert-base-uncased'
bertweet_model_name = 'vinai/bertweet-base'

# Download the DistilBERT tokenizer and model
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_name)
distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_model_name)

# Save the DistilBERT tokenizer and model locally
distilbert_tokenizer.save_pretrained('./distilbert-base-uncased')
distilbert_model.save_pretrained('./distilbert-base-uncased')

# Download the BERTweet tokenizer and model
bertweet_tokenizer = AutoTokenizer.from_pretrained(bertweet_model_name)
bertweet_model = AutoModelForSequenceClassification.from_pretrained(bertweet_model_name)

# Save the BERTweet tokenizer and model locally
bertweet_tokenizer.save_pretrained('./bertweet-base')
bertweet_model.save_pretrained('./bertweet-base')
