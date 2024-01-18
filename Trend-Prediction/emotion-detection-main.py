import os
import string
import torch
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from sklearn.utils import shuffle
from datasets import Dataset, DatasetDict, Features, Value,ClassLabel
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.nn.functional import cross_entropy
from transformers import pipeline
from re import sub
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import Trainer
os.environ['WANDB_DISABLED'] = 'true'


bs = 8 # batch size
lr = 3e-5
epochs = 3
w_decay = 0.01
num_labels = 35
# model_ckpt = "roberta-base"
# model_ckpt = "facebook/bart-base"
model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def remove_usernames(raw):
        return sub(r'@[^\s]*', '', raw)

# @staticmethod
def remove_punctuation(raw):
    return raw.translate(str.maketrans('', '', string.punctuation))

# @staticmethod
def remove_links(raw):
    return sub(r'https?://\S+', '', raw)

def remove_stopwords(raw):
    st = set(stopwords.words('english'))
    return " ".join([word for word in raw.split() if word not in st])


def remove_unnecessary_char(text): 
    text = remove_usernames(text)
    text = remove_punctuation(text)
    text = remove_links(text)
    text = remove_stopwords(text)
    return text

twitter_comments = pd.read_csv('/home/siu856533724/code/pc114/Social-Networks/Trend-Prediction/DataSet/csv/twitter_comments.csv', sep=";", encoding='cp1252')

input_text = []

print("#Removing unnecessary character from the user comments.....")
for comnt in twitter_comments['Comments']:
    # print(comnt)
    comnt = str(comnt)
    comnt = remove_usernames(comnt)
    comnt = remove_punctuation(comnt)
    comnt = remove_links(comnt)
    # print(comnt)
    # Can we set the input comment lenght which will be consider for sentiment analysis?
    if len(comnt) > 5:
        input_text.append(comnt)

print("Done!")

#--------------------------------
# Reading the 3 different datasets from the CSV files
dataset1 = pd.read_csv('/home/siu856533724/code/pc114/Social-Networks/Trend-Prediction/DataSet/csv/tweet_emotions.csv')
dataset1 = dataset1.drop(columns='tweet_id')
dataset2 = pd.read_csv('/home/siu856533724/code/pc114/Social-Networks/Trend-Prediction/DataSet/csv/tweet_emotions_1.csv')
dataset3 = pd.read_csv('/home/siu856533724/code/pc114/Social-Networks/Trend-Prediction/DataSet/csv/reddit_emotions.csv')

# Reading 35 different Emotions from the text file
class_names = []
with open('/home/siu856533724/code/pc114/Social-Networks/Trend-Prediction/DataSet/emotions.txt') as f:
    while True:
        line = f.readline()
        if not line:
            break
        # print(line.strip())
        class_names.append(line.strip())

mapping = {}
for i, line in enumerate(class_names):
    mapping[line] = i

print(mapping)
#-------------------------------
# Concating the 3 dataset into single one
data_frame = pd.concat([dataset1, dataset2, dataset3], ignore_index=True)
data_frame['text'] = data_frame['text'].apply(lambda x: remove_unnecessary_char(x))
data_frame['label'] = data_frame['label'].str.strip().map(mapping)
data_frame.drop(data_frame.columns[-1], axis=1, inplace=True)
#-------------------------------
data_frame = shuffle(data_frame)
train_size = 0.8
validate_size = 0.1
# Spliting the dataset into 3 parts Training, Test, and Validation
train, validate, test = np.split(data_frame, [int(train_size * len(data_frame)), int((validate_size + train_size) * len(data_frame))])

print('Dataset information:')
print(f'Training data: {train.shape}')
print(f'Validation data: {validate.shape}')
print(f'Test data: {test.shape}')

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
validate = test.reset_index(drop=True)

print(train['label'].unique())
print(test['label'].unique())
print(validate['label'].unique())

# Setting the feature
ft = Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})

# Combine Multiple Datasets 
emotions = DatasetDict({
    "train": Dataset.from_pandas(train, features=ft),
    "test": Dataset.from_pandas(test,features=ft),
    "validation": Dataset.from_pandas(validate,features=ft)
    })

# Convert a single DataFrame to a Dataset
# emotions = Dataset.from_pandas(train,features=ft)
train_ds = emotions["train"]

text = 'Tokenisation of text is a core task of NLP.'
# Load parameters of the tokeniser
# model_ckpt = "distilbert-base-uncased"
# model_ckpt = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
# Model to device for accessing the GPU if supported otherthan that device would be CPU
model.to(device)

print('#Info related to tokenizer!')
print(f'Vocab size: {tokenizer.vocab_size}')
print(f'Max length: {tokenizer.model_max_length}')
print(f'Tokeniser model input names: {tokenizer.model_input_names}')

print('Encoded text')
encoded_text = tokenizer(text)
print(encoded_text,'\n')

print('Tokens')
tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens,'\n')

print('#Convert tokens to string')
print(tokenizer.convert_tokens_to_string(tokens),'\n')

emotions.reset_format()

# Tokenisation function
def tokenise(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# apply to the entire dataset (train,test and validation dataset)
emotions_encoded = emotions.map(tokenise, batched=True, batch_size=None)
print(emotions_encoded["train"].column_names)

# text = "this is a test mukul"
# inputs = tokenizer(text, return_tensors="pt")
# print(inputs)
# print(f"Input tensor shape: {inputs['input_ids'].size()}")

# inputs = {k:v.to(device) for k,v in inputs.items()}

# with torch.no_grad():
#     outputs = model(**inputs)
# print(outputs)

# print(outputs.last_hidden_state.size())
# print(outputs.last_hidden_state[:,0].size())

# def extract_hidden_states(batch):
#     # Place model inputs on the GPU
#     inputs = {k:v.to(device) for k,v in batch.items()
#               if k in tokenizer.model_input_names}
    
#     # Extract last hidden states
#     with torch.no_grad():
#         last_hidden_state = model(**inputs).last_hidden_state
        
#     # Return vector for [CLS] token
#     return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Extract last hidden states (faster w/ GPU)
# emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)
# emotions_hidden["train"].column_names

# Getting Feature and Label data for Training and Validation
# X_train = np.array(emotions_hidden["train"]["hidden_state"])
# X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
# y_train = np.array(emotions_hidden["train"]["label"])
# y_valid = np.array(emotions_hidden["validation"]["label"])
# print(f'Training Dataset: {X_train.shape}')
# print(f'Validation Dataset {X_valid.shape}')

# Model for sequence classification
model = (AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device))

# For Measureing Model performance
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

logging_steps = len(emotions_encoded["train"]) // bs
model_name = f"{model_ckpt}-finetuned-emotion-model"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=epochs,      # number of training epochs
                                  learning_rate=lr,             # model learning rate
                                  per_device_train_batch_size=bs, # batch size
                                  per_device_eval_batch_size=bs,  # batch size
                                  weight_decay=w_decay,           # weight decay
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False, 
                                  report_to="none",
                                  logging_steps=logging_steps,
                                  push_to_hub=False,
                                  log_level="error")


trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)
trainer.train()

# Predict on Validation Dataset
pred_output = trainer.predict(emotions_encoded["test"])

print(f'Output Predition: {pred_output.predictions.shape}')
print(pred_output.predictions)

y_preds = np.argmax(pred_output.predictions,axis=1)
print(f'Output Prediction:{y_preds.shape}')
print(f'Predictions: {y_preds}')

# def forward_pass_with_label(batch):
    
#     # Place all input tensors on the same device as the model
#     inputs = {k:v.to(device) for k,v in batch.items()
#               if k in tokenizer.model_input_names}

#     with torch.no_grad():
#         output = model(**inputs)
#         pred_label = torch.argmax(output.logits, axis=-1)
#         loss = cross_entropy(output.logits, batch["label"].to(device), reduction="none")
        
#     # Place outputs on CPU for compatibility with other dataset columns
#     return {"loss": loss.cpu().numpy(), "predicted_label": pred_label.cpu().numpy()}

# # Convert our dataset back to PyTorch tensors
# emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
# # Compute loss values
# emotions_encoded["test"] = emotions_encoded["test"].map(forward_pass_with_label,
#                                                                     batched=True, 
#                                                                     batch_size=bs)

# load from previously saved model
trainer.save_model()
classifier = pipeline("text-classification", model=model_name)

# New unseen by model data
new_data = input(str('input the text')) # Example: 'I watched a movie last night, it was quite brilliant'

preds = classifier(new_data, return_all_scores=True)
df_preds = pd.DataFrame(preds[0])
print("Input text:")
print(new_data)
print("Predicted Score:")
print(df_preds)