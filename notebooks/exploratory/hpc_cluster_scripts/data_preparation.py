import pandas as pd

# Set the path to the dataset
dataset_path = '../../../datasets/TruthSeeker2023/Truth_Seeker_Model_Dataset.csv'

# Load the dataset into a pandas dataframe, ensuring the header is inferred from the first row
df = pd.read_csv(dataset_path, header=0)

# First, eliminate from the dataframe every row that has in the corresponding column the value of NO MAJORITY, or the value of unrelated.
df = df[(df['5_label_majority_answer'] != 'NO MAJORITY') & (df['3_label_majority_answer'] != 'unrelated')]

# Function to calculate categorical_label
def calculate_categorical_label(row):
    label = row['3_label_majority_answer'].lower()  # convert to lowercase
    if row['target'] == True and label == 'agree':
        return True
    elif row['target'] == True and label == 'disagree':
        return False
    elif row['target'] == False and label == 'agree':
        return False
    elif row['target'] == False and label == 'disagree':
        return True

# Apply the function to the dataframe
df['categorical_label'] = df.apply(calculate_categorical_label, axis=1)

# Create BinaryNumLabel column
df['BinaryNumLabel'] = df['categorical_label'].apply(lambda x: 1.0 if x == True else 0.0)

# Save the dataframe
df.to_csv('processed_tweets_dataset.csv', index=False)
