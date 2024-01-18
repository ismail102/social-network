from datasets import load_metric
import evaluate
import numpy as np
import torch.nn as nn
import torch
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from transformers import EvalPrediction

# class_labels = [i for i in range(len(mapping))]
# num_labels = len(mapping)

def loss_fn(outputs, labels):
    if labels is None:
        return None
    loss = nn.BCEWithLogitsLoss()
    # loss = nn.MultiLabelMarginLoss()
    return loss(outputs, labels.float())

def log_metrics(preds, labels):
    preds = torch.stack(preds)
    preds = preds.cpu().detach().numpy()
    labels = torch.stack(labels)
    labels = labels.cpu().detach().numpy()
    
    ## Method 2 using ravel()
    fpr_micro, tpr_micro, _ = metrics.roc_curve(labels.ravel(), preds.ravel())
    auc_micro = metrics.auc(fpr_micro, tpr_micro)

    # Assuming preds is a NumPy array
    threshold = 0.4
    # preds = np.where(np.logical_or(preds > threshold, preds == np.max(preds)), 1, 0)

    preds1 = []
    for pred in preds:
        preds1.append(np.where(np.logical_or(pred > threshold, pred == np.max(pred)), 1, 0))
    
    preds = np.array(preds1)
    # print("Prediction2: ", preds)
    # Accuracy
    accuracy = accuracy_score(labels.ravel(), preds.ravel())
    # Precision
    precision = precision_score(labels.ravel(), preds.ravel(), average='macro')  # You can change 'micro' to other options like 'macro' or 'weighted'
    # Recall
    recall = recall_score(labels.ravel(), preds.ravel(), average='macro')  # Change 'micro' as needed
    # F1 score
    f1 = f1_score(labels.ravel(), preds.ravel(), average='macro')
    
    return {"auc": auc_micro, "f1": f1, "recall": recall, "precision": precision, "acc": accuracy}

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def custom_metrics(logits, labels):
    metric1 = evaluate.load("precision")
    metric2 = evaluate.load("recall")
    metric3 = evaluate.load("f1")
    metric4 = evaluate.load("accuracy")
    
    # logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision = metric1.compute(predictions=predictions, references=labels, average='weighted')['precision']
    recall = metric2.compute(predictions=predictions, references=labels, average='weighted')['recall']
    f1 = metric3.compute(predictions=predictions, references=labels, average='weighted')['f1']
    accuracy = metric4.compute(predictions=predictions, references=labels)['accuracy']

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

def get_emotions():
    mapping = {
        0:"admiration",
        1:"amusement",
        2:"anger",
        3:"annoyance",
        4:"approval",
        5:"caring",
        6:"confusion",
        7:"curiosity",
        8:"desire",
        9:"disappointment",
        10:"disapproval",
        11:"disgust",
        12:"embarrassment",
        13:"excitement",
        14:"fear",
        15:"gratitude",
        16:"grief",
        17:"joy",
        18:"love",
        19:"nervousness",
        20:"optimism",
        21:"pride",
        22:"realization",
        23:"relief",
        24:"remorse",
        25:"sadness",
        26:"surprise",
        27:"neutral"
    }
    return mapping

def get_emotions_dict():
    mapping = {
        '0':"admiration",
        '1':"amusement",
        '2':"anger",
        '3':"annoyance",
        '4':"approval",
        '5':"caring",
        '6':"confusion",
        '7':"curiosity",
        '8':"desire",
        '9':"disappointment",
        '10':"disapproval",
        '11':"disgust",
        '12':"embarrassment",
        '13':"excitement",
        '14':"fear",
        '15':"gratitude",
        '16':"grief",
        '17':"joy",
        '18':"love",
        '19':"nervousness",
        '20':"optimism",
        '21':"pride",
        '22':"realization",
        '23':"relief",
        '24':"remorse",
        '25':"sadness",
        '26':"surprise",
        '27':"neutral"
    }
    return mapping

def get_sentiment_dict():
    sentiment_dict = {
        "positive": ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"],
        "negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"],
        "ambiguous": ["realization", "surprise", "curiosity", "confusion"],
        "neutral": ["neutral"]
        }
    return sentiment_dict

def get_trends():
    trends = ["approval","toxic","obscene", 'insult', "threat", "hate", "offensive", "neither"]
    return trends

if __name__ == '__main__':
    print("#Common Function")