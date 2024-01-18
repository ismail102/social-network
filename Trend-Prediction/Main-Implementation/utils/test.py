
import torch
import torch
import torch.nn as nn
from tqdm.notebook import tqdm

def loss_fn(y_pred, y_true):
    
    if y_true is None:
        return None
    
    # if not all others
    return nn.BCEWithLogitsLoss()(y_pred, y_true.float())


def test_fn(data_loader, model, device):
    test_loss = 0.0
    model.eval()
    fin_targets = []
    fin_outputs = []

    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            mask = d["mask"]
            # targets = d["labels"]
            # token_type_ids = d['token_type_ids']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            # targets = targets.to(device, dtype=torch.float)

            outputs = model(ids, attention_mask=mask)
            # loss = loss_fn(y_pred=outputs.logits, y_true=targets)
#             logits = outputs.logits
            
            # test_loss += loss.item()
            
            # fin_targets.extend(targets)
            fin_outputs.extend(torch.nn.functional.softmax(outputs.logits))

    return fin_outputs

if __name__ == '__main__':
     print("#Data testing...")
