
import os
import torch
from transformers import AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import glob

def ret_model(model_ckpt, n_labels, is_freez):
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=n_labels)

    if model_ckpt != 'roberta-large' and is_freez:
        # Freeze all layers except the classification layer
        for param in model.base_model.parameters():
            param.requires_grad = False

        # Modify the classification layer for the new task (target domain)
        model.classifier = torch.nn.Linear(model.classifier.in_features, n_labels)

    return model


def get_model_tokenizer(pred_type, mode_pt_file, model_ckpt, labels, is_freez, gpu):

    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')

    parent_dir = f'/home/siu856533724/code/source-code/Social-Networks/Trend-Prediction/Main-Implementation/model_save/{pred_type}/BERT'
    # def get_files_by_extension(folder_path, extension):
    #     pattern = folder_path + f"/*.{extension}"
    #     files = glob.glob(pattern)
    #     return files
    model = ret_model(model_ckpt, labels, is_freez)
    tokenizer = AutoTokenizer.from_pretrained(parent_dir, do_lower_case=True)

    model.to(device)

    model_dir = os.path.join(parent_dir, mode_pt_file) #get_files_by_extension(parent_dir, extension)
    # model_dir = os.path.join(parent_dir, "*.pt")
    model = torch.load(model_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model = model_to_save
    
    return model, tokenizer


# Tokenize all of the sentences and map the tokens to thier word IDs.

def Tokenize(text, max_length, tokenizer):
    encoded_dict = tokenizer.encode_plus(
                        str(text),              # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_length,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True, # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                        truncation = True
                    )
    # Add the encoded sentence to the list.    
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    # Convert the lists into tensors.
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)

    return {"input_ids": input_ids, "attention_mask": attention_mask}

def Classification(dict, model, gpu):
    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    ids = dict['input_ids'].to(device, dtype=torch.long)
    mask = dict['attention_mask'].to(device, dtype=torch.long)

    outputs = model(ids, attention_mask=mask)
    result = torch.nn.functional.softmax(outputs.logits)
    result = result.cpu().data.numpy()
    scores = result.tolist()
    return scores

if __name__ == '__main__':
    print("#bert-classification")