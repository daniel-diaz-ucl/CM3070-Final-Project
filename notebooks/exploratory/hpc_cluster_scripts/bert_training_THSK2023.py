import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (BertForSequenceClassification, BertTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

# Load the processed dataframe
#df = pd.read_csv('processed_tweets_dataset.csv')
df = pd.read_pickle('processed_tweets_dataset.pkl')

# Define the TweetDataset class
class TweetDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer, max_len):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'tweet_text': tweet,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load pre-trained model and tokenizer from local directory
model_name = './bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Specify the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Preprocess the tweets
def preprocess(tweet):
    encoding = tokenizer.encode_plus(
        tweet,
        add_special_tokens=True,
        max_length=512,  # Adjust as needed
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()

# Apply preprocessing to the entire dataset
X = df['tweet'].apply(preprocess)
y = df['BinaryNumLabel'].tolist()

# Split the data into training+validation and testing sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training+validation set into separate training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Create the TweetDataset
train_dataset = TweetDataset(
    tweets=X_train.tolist(),
    labels=y_train,
    tokenizer=tokenizer,
    max_len=512  # Adjust as needed
)

val_dataset = TweetDataset(
    tweets=X_val.tolist(),
    labels=y_val,
    tokenizer=tokenizer,
    max_len=512  # Adjust as needed
)

test_dataset = TweetDataset(
    tweets=X_test.tolist(),
    labels=y_test,
    tokenizer=tokenizer,
    max_len=512  # Adjust as needed
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=32,  # Adjust as needed
    per_device_eval_batch_size=32,  # Adjust as needed
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    learning_rate=1e-5,
    weight_decay=0.0,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
)

# Compute metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train and evaluate
trainer.train()
trainer.evaluate()

# Save model
model.save_pretrained('./best_model')
tokenizer.save_pretrained('./best_model')

# Plot loss
loss_values = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('training_loss.png')

# Heatmap
def plot_heatmap(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

# Evaluate on test set
predictions = trainer.predict(test_dataset)
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)
plot_heatmap(y_true, y_pred)
