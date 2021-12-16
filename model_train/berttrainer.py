# -*- ecoding: utf-8 -*-
# @ModuleName: trainer
# @Function: 
# @Author: long
# @Time: 2021/12/13 20:25
# *****************conding***************
import numpy as np
import torch.optim as optim
import torch.nn.functional
from torch.utils.data import Dataset
import torch
import random
from transformers import BertTokenizer
import json
from data_process.mapidrel import map_id_rel
from models import BERT_Classifier


def load_train():
    rel2id, id2rel = map_id_rel()
    max_length = 200  # 在训练数据和预测数据中都制定了相关的max-len用于实现相应的裁剪操作
    tokenizer = BertTokenizer.from_pretrained('./model_train/bert-base-model/')
    train_data = {}
    train_data['label'] = []
    train_data['mask'] = []
    train_data['text'] = []  # 需要从数据库中读取相应的数据

    with open("./data/dev.json", 'r', encoding='utf-8') as load_f:
        temp = load_f.readlines()
        temp = temp[:]  # 这里的长度应该在后续的训练过程中改成20000条数据
        for line in temp:
            dic = json.loads(line)
            if dic['rel'] not in rel2id:
                train_data['label'].append(0)
            else:
                train_data['label'].append(rel2id[dic['rel']])
            sent = dic['ent1'] + dic['ent2'] + dic['text']
            indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
            avai_len = len(indexed_tokens)
            while len(indexed_tokens) < max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[: max_length]
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

            # Attention mask
            att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
            att_mask[0, :avai_len] = 1
            train_data['text'].append(indexed_tokens)
            train_data['mask'].append(att_mask)
    return train_data


def load_dev():
    rel2id, id2rel = map_id_rel()
    max_length = 200  #
    tokenizer = BertTokenizer.from_pretrained('./model_train/bert-base-model/')
    train_data = {}
    train_data['label'] = []
    train_data['mask'] = []
    train_data['text'] = []

    with open("./data/dev.json", 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            dic = json.loads(line)
            if dic['rel'] not in rel2id:
                train_data['label'].append(0)
            else:
                train_data['label'].append(rel2id[dic['rel']])

            sent = dic['ent1'] + dic['ent2'] + dic['text']
            indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
            avai_len = len(indexed_tokens)
            while len(indexed_tokens) < max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[: max_length]
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

            # Attention mask
            att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
            att_mask[0, :avai_len] = 1
            train_data['text'].append(indexed_tokens)
            train_data['mask'].append(att_mask)
    return train_data


# prepare_data()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def berttraindata():
    setup_seed(44)

    rel2id, id2rel = map_id_rel()

    print(len(rel2id))
    print(id2rel)

    USE_CUDA = torch.cuda.is_available()
    # USE_CUDA=False

    data = load_train()
    train_text = data['text']
    train_mask = data['mask']
    train_label = data['label']

    train_text = [t.numpy() for t in train_text]
    train_mask = [t.numpy() for t in train_mask]

    train_text = torch.tensor(train_text)
    train_mask = torch.tensor(train_mask)
    train_label = torch.tensor(train_label)

    print("--train data--")
    print(train_text.shape)
    print(train_mask.shape)
    print(train_label.shape)

    data = load_dev()
    dev_text = data['text']
    dev_mask = data['mask']
    dev_label = data['label']

    dev_text = [t.numpy() for t in dev_text]
    dev_mask = [t.numpy() for t in dev_mask]

    dev_text = torch.tensor(dev_text)
    dev_mask = torch.tensor(dev_mask)
    dev_label = torch.tensor(dev_label)

    print("--train data--")
    print(train_text.shape)
    print(train_mask.shape)
    print(train_label.shape)

    print("--eval data--")
    print(dev_text.shape)
    print(dev_mask.shape)
    print(dev_label.shape)

    # exit()
    # USE_CUDA=False

    if USE_CUDA:
        print("using GPU")

    train_dataset = torch.utils.data.TensorDataset(train_text, train_mask, train_label)
    dev_dataset = torch.utils.data.TensorDataset(dev_text, dev_mask, dev_label)


    def get_model():
        labels_num = len(rel2id)
        # from BERT_Pytorch_Fintuing.model import BERT_Classifier
        model = BERT_Classifier(labels_num)
        return model


    def eval(net, dataset, batch_size):
        net.eval()
        train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
        with torch.no_grad():
            correct = 0
            total = 0
            iter = 0
            for text, mask, y in train_iter:
                iter += 1
                if text.size(0) != batch_size:
                    break
                text = text.reshape(batch_size, -1)
                mask = mask.reshape(batch_size, -1)

                if USE_CUDA:
                    text = text.cuda()
                    mask = mask.cuda()
                    y = y.cuda()

                outputs = net(text, mask, y)
                # print(y)
                loss, logits = outputs[0], outputs[1]
                _, predicted = torch.max(logits.data, 1)
                total += text.size(0)
                correct += predicted.data.eq(y.data).cpu().sum()
                s = ("Acc:%.3f" % ((1.0 * correct.numpy()) / total))
            acc = (1.0 * correct) / total
            print("Eval Result: right", correct.cpu().numpy().tolist(), "total", total, "Acc:", acc)
            return acc

    def train(net, dataset, num_epochs, learning_rate, batch_size):
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0)
        # optimizer = AdamW(net.parameters(), lr=learning_rate)
        train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
        pre = 0
        for epoch in range(num_epochs):
            correct = 0
            total = 0
            iter = 0
            for text, mask, y in train_iter:
                iter += 1
                optimizer.zero_grad()
                # print(type(y))
                # print(y)
                if text.size(0) != batch_size:
                    break
                text = text.reshape(batch_size, -1)
                mask = mask.reshape(batch_size, -1)
                if USE_CUDA:
                    text = text.cuda()
                    mask = mask.cuda()
                    y = y.cuda()
                # print(text.shape)
                loss, logits = net(text, mask, y)
                # print(y)
                # print(loss.shape)
                # print("predicted",predicted)
                # print("answer", y)
                loss.backward()
                optimizer.step()
                # print(outputs[1].shape)
                # print(output)
                # print(outputs[1])
                _, predicted = torch.max(logits.data, 1)
                total += text.size(0)
                correct += predicted.data.eq(y.data).cpu().sum()
            loss = loss.detach().cpu()
            print("epoch ", str(epoch), " loss: ", loss.mean().numpy().tolist(), "right", correct.cpu().numpy().tolist(),
                  "total", total, "Acc:", correct.cpu().numpy().tolist() / total)
            acc = eval(model, dev_dataset, 8)
            if acc > pre:
                pre = acc
                torch.save(model, './best_model_save/bert_modelsave_best' + '.pth')
        return 0
    model = get_model()
    # model=nn.DataParallel(model,device_ids=[0,1])
    if USE_CUDA:
        model = model.cuda()
    # eval(model,dev_dataset,8)
    train(model, train_dataset, 20, 0.001, 8)  # batch_size可以设置的稍微大一些，以便后续的调用，此时的batch-size大小应该为4
    # eval(model,dev_dataset,8)
    return "trained"

