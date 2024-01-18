import torch
from utils.common_functions import loss_fn
from tqdm import tqdm
import torch.nn as nn

from utils.common_functions import custom_metrics, flat_accuracy, log_metrics
def model_eval(dataloader, model, device):

      # '''
    #     Modified from Abhishek Thakur's BERT example: 
    #     https://github.com/abhishekkrthakur/bert-sentiment/blob/master/src/engine.py
    # '''
    
    criterion = nn.CrossEntropyLoss().to(device)
    eval_loss = 0.0
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in enumerate(tqdm(dataloader)):
            ids = d["input_ids"]
            token_type_ids = d['token_type_ids']
            mask = d["attention_mask"]
            targets = d["labels"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = criterion(outputs, targets)
            # loss = loss_fn(outputs, targets)
            eval_loss += loss.item()
            fin_targets.extend(targets)
            fin_outputs.extend(torch.sigmoid(outputs))
    return eval_loss, fin_outputs, fin_targets


if __name__ == '__main__':
    print("#Evaluate Function")