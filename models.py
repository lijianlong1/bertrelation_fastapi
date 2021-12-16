# -*- ecoding: utf-8 -*-
# @ModuleName: models
# @Function: 
# @Author: long
# @Time: 2021/12/14 9:26
# *****************conding***************
import torch.nn as nn
from transformers import BertModel
class BERT_Classifier(nn.Module):
    def __init__(self, label_num):
        super().__init__()
        self.encoder = BertModel.from_pretrained('./model_train/bert-base-model/')  # model需要重新的引入
        self.dropout = nn.Dropout(0.1, inplace=False)
        self.fc = nn.Linear(768, label_num)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x, mask, label=None):
        x = self.encoder(x, attention_mask=mask)[0]
        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.fc(x)
        if label == None:
            return None, x
        else:
            return self.criterion(x, label), x

class trans(nn.Module):
    def __init__(self, label_num):
        super().__init__()
        self.encoder = nn.Embedding(num_embeddings=21128,embedding_dim=128)#BertModel.from_pretrained('./model_train/bert-base-model/')  # model需要重新的引入
        self.dropout = nn.Dropout(0.1, inplace=False)
        self.bigru = nn.GRU(128, 128,
                            batch_first=True,
                            bidirectional=True)  #
        self.fc = nn.Linear(256, label_num)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, mask, label=None):
        x = self.encoder(x)
        #print('embedding',x.shape)
        x = self.dropout(x)
        x, (h_n, c_n) = self.bigru(x)
        #print('lstm', x.shape)
        x = x[:, -1, :] # 取lstm的最后一个
        x = self.fc(x)
        if label == None:
            return None, x
        else:
            return self.criterion(x, label), x