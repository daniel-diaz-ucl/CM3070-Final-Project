from transformers import (BertForSequenceClassification, BertTokenizer,
                          RobertaForSequenceClassification, RobertaTokenizer)

# Specify the model names
#bert_model_name = 'bert-large-uncased'
roberta_model_name = 'roberta-large'

# Download the BERT tokenizer and model
#bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
#bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)

# Save the BERT tokenizer and model locally
#bert_tokenizer.save_pretrained('./bert-large-uncased')
#bert_model.save_pretrained('./bert-large-uncased')

# Download the RoBERTa tokenizer and model
roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_name)
roberta_model = RobertaForSequenceClassification.from_pretrained(roberta_model_name)

# Save the RoBERTa tokenizer and model locally
roberta_tokenizer.save_pretrained('./roberta-large')
roberta_model.save_pretrained('./roberta-large')
