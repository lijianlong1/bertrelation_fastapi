# -*- ecoding: utf-8 -*-
# @ModuleName: model
# @Function: 
# @Author: long
# @Time: 2021/12/13 20:24
# *****************conding***************

import pymysql
from transformers import BertTokenizer
import torch
import settings
import os

MYSQL_EVEN = settings.MYSQL_SETTING

#print(host)
host = MYSQL_EVEN['default']['hosts']
port = MYSQL_EVEN['default']['port']
username = MYSQL_EVEN['default']['username']
password = MYSQL_EVEN['default']['password']
dbname = MYSQL_EVEN['default']['dbname']

def map_id_rel():
    id2rel = {0: 'UNK', 1: '主演', 2: '歌手', 3: '简称', 4: '总部地点', 5: '导演', 6: '出生地', 7: '目', 8: '出生日期', 9: '占地面积',
              10: '上映时间', 11: '出版社', 12: '作者', 13: '号', 14: '父亲', 15: '毕业院校', 16: '成立日期', 17: '改编自', 18: '主持人',
              19: '所属专辑', 20: '连载网站', 21: '作词', 22: '作曲', 23: '创始人', 24: '丈夫', 25: '妻子', 26: '朝代', 27: '民族', 28: '国籍',
              29: '身高', 30: '出品公司', 31: '母亲', 32: '编剧', 33: '首都', 34: '面积', 35: '祖籍', 36: '嘉宾', 37: '字', 38: '海拔',
              39: '注册资本', 40: '制片人', 41: '董事长', 42: '所在城市', 43: '气候', 44: '人口数量', 45: '邮政编码', 46: '主角', 47: '官方语言',
              48: '修业年限'}
    # 在此需要改变相应的数据格式
    rel2id = {}
    for i in id2rel:
        rel2id[id2rel[i]] = i
    return rel2id, id2rel

def map_id_rel_from_sql():
    db = pymysql.connect(user=username,password=password,host=host, port=port,database=dbname)
    cursor = db.cursor()
    sql = "select relationship_content from train_data"
    # 执行SQL语句
    cursor.execute(sql)
    # 获取所有记录列表
    results = cursor.fetchall()
    data_train_all = []
    id2rel = {}
    rel_list = ['UNK']
    for row in results:
        data_each = list(eval(row[0]))
        # print(data_each)
        if data_each:  # 判断标记数据是否为空
            data_train_all.append({"rel":data_each[0]['rel'],"ent1":data_each[0]['ent1'],
                                   "ent2":data_each[0]['ent2'],"text":data_each[0]['text']})
            if data_each[0]['rel'] not in rel_list:
                rel_list.append(data_each[0]['rel'])  # 至此得到所有的关系标签
    # 在这个地方进行标签的制作
    for index, i in enumerate(rel_list,0):
        id2rel[index] = i
    # 至此数据库中的标签数据已经拿出
    #print(data_train_all)
    rel2id = {}
    for i in id2rel:
        rel2id[id2rel[i]] = i
    # 这个地方所有的关系都也已经拿到，需要对获取的数据进行操作
    max_length = 64  # 在训练数据和预测数据中都制定了相关的max-len用于实现相应的裁剪操作
    tokenizer = BertTokenizer.from_pretrained('./model_train/bert-base-model/vocab.txt')
    train_data = {}
    train_data['label'] = []
    train_data['mask'] = []
    train_data['text'] = []
    for data in data_train_all:
        #data = json.loads(data)  # 拿到所有数据的
        #print(type(data))
        dic = data
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
    db.close()
    return rel2id, id2rel,train_data
    # 关闭数据库连接

# rel2id, id2rel,train_data = map_id_rel_from_sql()
# print(rel2id, id2rel,train_data)
# 保留相应的关系预测的接口，实现关系的添加处理
def model_version():
    """
    查询模型的版本号
    :return:模型的相关信息
    """
    db = pymysql.connect(user=username, password=password, host=host, port=port, database=dbname)
    cursor = db.cursor()
    sql = "select * from train_model where model_is_use=1"
    cursor.execute(sql)
    # 获取所有记录列表
    results = cursor.fetchall()
    db.close()
    # print(results)
    model_message = {'id': results[0][0], 'model_type': results[0][1], 'model_name': results[0][2], 'model_desc': results[0][3],
                     'model_path': results[0][4], 'model_version_id': results[0][5], 'model_is_use':results[0][6]}

    # print(model_message)
    return model_message
# model_version()

def get_and_public_model():
    """
    # 从文件路径中查询模型的最大版本号信息，并返回最大的版本号的信息,发布模型还得查看，相应的数据中有没有相关的模型版本号，如果数据
    表为空，需要另行处理
    :return:
    """
    model_file_dir = './best_model_save'
    model_version_list = []
    for root, file_dir, filename in os.walk(model_file_dir):
        #print(filename)
        # 此时的filename，全部已经拿到，需要匹配最大的模型id号
        for model_name in filename:
            id_each = model_name.split('_')[1].split('.')[0]
            #print(id_each)
            model_version_list.append(id_each)
    model_version_list.sort()
    max_model_version = model_version_list[-1]
    # print(max_model_version_id) 此时模型的最大版本号已经取得，需要进行数据库的相应匹配
    # 判断数据库中是否存在相应的模型版本
    #     # print(model_version())
    model_is_use_version = model_version()['model_version_id']
    if int(model_is_use_version) < int(max_model_version):
        model_name = '模型'+str(max_model_version)
        db = pymysql.connect(user=username, password=password, host=host, port=port, database=dbname)
        cursor = db.cursor()
        sql1 = "UPDATE train_model SET model_is_use=0 where model_is_use=1"
        sql2 = "INSERT INTO train_model(model_name,model_desc,model_type,model_version_id,model_is_use) VALUES ('关系抽取模型','模型的描述','relation——关系抽取',%f ,1)" % float(max_model_version)
        # 1更新数据
        try:
            cursor.execute(sql1)
            cursor.execute(sql2)
            db.commit()  # 提交数据
        except:
            print("数据操作失败")
            db.rollback()  # 数据更新失败时，进行回滚操作
        db.close()  # 关闭数据库连接
        return 'succeed publish'
    else:
        return '还没训练出新的模型，当前模型就为最大版本号'

def delete_max_version():
    """
    删除数据库和文件路径下的最大的模型
    :return:操作成功或者失败提示
    """
    # 还需要判断的条件有，假如数据库中，没有了相应的模型，则不进行删除，提示数据库中没有对应的模型版本
    model_is_use_version = model_version()['model_version_id']
    # 应该禁止删除一代模型

    print(model_is_use_version)
    if int(model_is_use_version) == 1:
        return '当前模型为最初模型，无法回退'# 一代模型为缓存，不允许删除
    else:
    #  获得模型的最大版本号
        db = pymysql.connect(user=username, password=password, host=host, port=port, database=dbname)
        cursor1 = db.cursor()

        sql1 = 'delete FROM train_model WHERE model_is_use=1'  # 删除
        sql2 = 'select max(model_version_id) model_version_id from train_model'  # 查询当前模型中的最大的模型版本号
        try:
            cursor1.execute(sql1)
            # 数据库删了相关的模型，
            cursor1.execute(sql2)
            result = cursor1.fetchall()
            print(result[0][0])
            sql3 = 'UPDATE train_model set model_is_use=1 where model_version_id=%f' % float(result[0][0])  # 将数据库中的模型最大版本号的model_is_use_置为1，操作慎用
            cursor1.execute(sql3)
            db.commit()  # 对数据库的操作一定记得提交
            # 删除文件中的GRU模型
            model_file_path = './best_model_save/GRU_'+str(int(model_is_use_version))+'.pth'
            os.remove(model_file_path)
        except Exception as e:
            print('模型删除失败',e)
            db.rollback()
        db.close()
        return 'succeed delete the model'

