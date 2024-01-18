import torch
from utils.common_functions import loss_fn
from tqdm import tqdm
import torch.nn as nn

def model_train(dataloader, model, optimizer, device, scheduler):
    # ========================================
    #               Training
    # ========================================
    criterion = nn.CrossEntropyLoss().to(device)
    train_loss = 0.0
    model.train()
    for bi, d in enumerate(tqdm(dataloader)):
        ids = d["input_ids"]
        token_type_ids = d['token_type_ids']
        mask = d["attention_mask"]
        targets = d["labels"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        torch.cuda.empty_cache()

        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = criterion(outputs, targets)
        # loss = loss_fn(outputs, targets)
        train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
    return train_loss


if __name__ == '__main__':
    print("#Train Function")