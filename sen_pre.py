# -*- ecoding: utf-8 -*-
# @ModuleName: sen_pre
# @Function: 
# @Author: long
# @Time: 2021/12/13 20:24
# *****************conding***************
import torch
from data_process.mapidrel import map_id_rel,map_id_rel_from_sql,model_version
from transformers import BertTokenizer
# from models import trans, BERT_Classifier
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

#
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
        print(x)
        if label == None:
            return None, x
        else:
            return self.criterion(x, label), x


def relation_pre(ent1,ent2,sen1,re_model="GRU"):
    """
    对相应的关系进行预测的接口，需要考虑到的有：是否将句子中的实体进行槽填充等操作，将句子中的实体删除，以免影响关系预测结果
    :param ent1: 传入的第一个实体
    :param ent2: 传入的第二个实体
    :param sen1: 传入的相关句子
    :param re_model: 表示使用的模型类型预测，默认使用GRU模型进行预测
    :return: {'ent1','ent2','sen1','rel_pre'}
    """
    model_is_use_version = model_version()['model_version_id']
    print(model_is_use_version)
    sen_concat = str(ent1) + str(ent2) + str(sen1)  # 至此，我们已经拿到了相应的句子连接
    #rel2id, id2rel = map_id_rel()
    rel2id, id2rel, _ =map_id_rel_from_sql()
    #vocab_file_path = ''
    tokenizer = BertTokenizer.from_pretrained('./model_train/bert-base-model/vocab.txt')
    indexed_tokens = tokenizer.encode(sen_concat, add_special_tokens=True)
    avai_len = len(indexed_tokens)
    if re_model == 'GRU':
        max_len = 64
        while len(indexed_tokens)<max_len:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:max_len]  # 此时拿到的数据长度为64
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # 将相关的数据进行压缩，此时的维度为（1，max_len）
        # mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, max_len)
        att_mask[0, :avai_len] = 1
        # 输入到模型中的相关数据
        data_one_text_input = indexed_tokens
        # print(data_one_text_input.shape)
        data_one_mask_input = att_mask
        data_one_label_input = 0
        use_model_path = './best_model_save/GRU_'+str(int(model_is_use_version))+'.pth'
        relation_pre = model_pre_relation(data_one_text_input, data_one_mask_input, data_one_label_input,use_model_path,len(rel2id), re_model='GRU')
        return_data = {'ent1': ent1, 'ent2': ent2, 'sen': sen1,
                       'relation': id2rel[relation_pre.item()]}
        return return_data
    if re_model == 'BERT':
        max_len = 200
        while len(indexed_tokens)<max_len:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:max_len]  # 此时拿到的数据长度为64
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # 将相关的数据进行压缩，此时的维度为（1，max_len）
        # mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, max_len)
        att_mask[0, :avai_len] = 1
        # 输入到模型中的相关数据
        data_one_text_input = indexed_tokens
        # print(data_one_text_input.shape)
        data_one_mask_input = att_mask
        data_one_label_input = 0
        relation_pre = model_pre_relation(data_one_text_input, data_one_mask_input, data_one_label_input,
                                          './best_model_save/bert_modelsave_best.pth', len(rel2id), re_model='BERT')
        return_data = {'ent1': ent1, 'ent2': ent2, 'sen': sen1,
                       'relation': id2rel[relation_pre.item()]}
        return return_data

def model_pre_relation(text, att_mask, data_label,model_path,num_label,re_model='GRU'):
    """
    :param text:输入文本
    :param att_mask: mask
    :param data_label: 随便的一个标签向量
    :param model_path: 模型的存储路径
    :param num_label: 模型的关系种类
    :param re_model: 默认使用GRU模型进行训练和测试
    :return: predicted
    """
    if re_model == "GRU":
        with torch.no_grad():
            model = torch.load(model_path)
            # model = model.load_state_dict(torch.load(model_path,map_location='cpu'))
            outputs = model(text, att_mask)  # 此时的label按照unknow进行处理。其实这一个模型的不应该在此出现ylabel的值
            # print(outputs)
            loss, logits = outputs[0], outputs[1]
            _, predicted = torch.max(logits.data, 1)
            return predicted
    if re_model == 'BERT':
        with torch.no_grad():
            model = torch.load(model_path)
            #model = model.modules()
            outputs = model(text, att_mask)  # 此时的label按照unknow进行处理。其实这一个模型的不应该在此出现ylabel的值
            # print(outputs)
            loss, logits = outputs[0], outputs[1]
            _, predicted = torch.max(logits.data, 1)
            return predicted
    # 就直接将所有的模型放到cpu上运行






