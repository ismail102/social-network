from transformers import AutoModel
import torch.nn as nn
import torch

class TextClassification(nn.Module):
    def __init__ (self, n_classes, dropout, model_ckpt):
        super(TextClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(model_ckpt)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(768, n_classes)
        
    def forward(self, ids, mask, token_type_ids):
        pooledOut = self.bert(ids, attention_mask = mask, token_type_ids = token_type_ids)
        dropOut = self.dropout(pooledOut[1])
        output = self.out(dropOut)
        
        return output
    
class TextClassificationBertLSTM(nn.Module):
    def __init__ (self, n_classes, model_ckpt):
        super(TextClassificationBertLSTM, self).__init__()
        self.bert = AutoModel.from_pretrained(model_ckpt)
        # self.dropout = nn.Dropout(config.dropout)
        # self.fc = nn.Linear(1024, 768)
        # self.dropout2 = nn.Dropout(config.dropout)
        # self.out = nn.Linear(768, n_classes)

        ### New layers:
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(256*2, n_classes)
        
    def forward(self, ids, mask, token_type_ids):
        pooledOut = self.bert(ids, attention_mask = mask, token_type_ids = token_type_ids)
        # dropOut = self.dropout(pooledOut[1])
        # fcOut = self.fc(dropOut)
        # dropOut = self.dropout2(fcOut)
        # output = self.out(dropOut)

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        lstm_output, (h,c) = self.lstm(pooledOut[0]) ## extract the 1st token's embeddings
        hidden = torch.cat((lstm_output[:,-1, :256], lstm_output[:,0, 256:]), dim=-1)
        # dropOut = self.dropout(hidden[:, -1, :])
        output = self.linear(hidden.view(-1,256*2)) ### assuming that you are only using the output of the last LSTM cell to perform classification
        
        return output