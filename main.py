# -*- ecoding: utf-8 -*-
# @ModuleName: main
# @Function: 
# @Author: long
# @Time: 2021/12/13 20:16
# *****************conding***************
from fastapi import FastAPI
from pydantic import BaseModel
from model_train.berttrainer import berttraindata
from model_train.transformertrainer import transtraindata
from sen_pre import relation_pre
from data_process.mapidrel import model_version,get_and_public_model,delete_max_version
import warnings
warnings.filterwarnings('ignore')

class Sen(BaseModel):  #此类表示一个输入数据的相应组成形式
    ent1:str
    ent2:str
    sen1:str

app = FastAPI()


@app.get("/bertrelationtrain/") # 此接口为事件抽取接口
async def bertrelationtrain():
    """
    # 接口1：BERT模型训练接口
    :return: 训练成功提示
    """
    # 需要发送train操作的密码，才能进行train的使用
    #sentence = sen.data  # 至此相关的数据都已经拿到，拿到的数据格式为：{'data':"XXXXX"}
    # 编写一个函数，实现第二种事件抽取方案，使用传统的事件抽取思路进行模型的搭建和相关的测试。
    #  def extraction2(): 实现事件抽取功能
    # 生成一个版本的模型
    message = berttraindata()
    # 为生成的模型修改名称，使用几点几的方式进行
    return message

@app.get("/transformerrelationtrain/") # 此接口为事件抽取接口
async def transformerrelationtrain():
    """
    # 接口2：GRU模型训练接口(训练时间短，模型准确率高),
    ## 注意：模型训练完成后，一定要点发布按钮（训练完成后不能点击删除相关的模型）
    :return: 模型提示信息
    """
    # 需要发送train操作的密码，才能进行train的使用
    # sentence = sen.data  # 至此相关的数据都已经拿到，拿到的数据格式为：{'data':"XXXXX"}
    # 编写一个函数，实现第二种事件抽取方案，使用传统的事件抽取思路进行模型的搭建和相关的测试。
    #  def extraction2(): 实现事件抽取功能
    message = transtraindata()
    return message

@app.post("/relationpre")
async def pre(sen: Sen):
    """
    # 接口3：关系识别接口
    :param sen: 传入的数据
    :return:
    """
    ent1 = sen.ent1
    ent2 = sen.ent2
    sen1 = sen.sen1
    # 编写一个相应的关系识别的接口，直接使用相应的函数进行操作
    result = relation_pre(ent1, ent2, sen1, re_model='GRU')
    return result


@app.get("/getmodel_version/") #
async def get_model_v():
    """
    # 接口4：获取模型版本号,和相应的模型所有信息
    此接口为：获取模型的版本号的一个接口
    :return: 模型的版本信息
    """
    result = model_version()
    version = result['model_version_id']
    return result, version

@app.get("/model_public/") # 模型版本发布
async def model_public():
    """
    # 接口5：模型版本发布（将生成的最大模型版本号存入到数据库中）
    函数功能：将数据库中的train_model表加入相应的数据，模型的版本号按照整数形式进行更新迭代
    控制逻辑为：
    1.从数据库中获取当前模型的版本信息
    2.从训练好的模型中获取当前最大的模型版本号
    3.执行数据库的操作，将当前数据库中的train_model表进行操作和处理
    获取所有模型数据，将所有的模型的is_use设置为0，然后再遍历文件夹，获取模型最大的version
    如果version>is_use_version,则将模型的信息存入数据库中
    :return: message of success or defeat
    """
    message_succeed = get_and_public_model()
    return message_succeed

@app.get("/delete_model/") # 模型版本发布
async def model_delete():
    """
    # 接口6：删除最大版本号的模型（新训练的模型效果差，没必要进行保存操作）
    ### 1.查询模型中的最大版本号的模型，将文件夹下的模型删除，并删除数据库中的模型，然后将当前的最大版本号的模型的is_model_use更改为1
    :return:删除成功提示
    """
    # 操作数据库
    operate = delete_max_version()
    return operate



