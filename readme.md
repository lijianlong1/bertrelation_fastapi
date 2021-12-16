# 自动训练和关系识别接口
## 使用方式
### 1.pip安装相关的依赖文件
### 2.到根目录下执行(指定端口号执行)：uvicorn main:app --host 0.0.0.0 --port 8080 --reload
## 接口管理于接口的调用方式
### 访问：运行端口号下的docs页面，127.0.0.1:8080/docs
## 接口说明（提供六个接口）目前BERT模型的还没整理
### 接口1：BERT模型训练接口
### 接口2：GRU模型训练接口(训练时间短，模型准确率高)
### 接口3：关系识别接口
### 接口4：获取模型版本号,和相应的模型所有信息
### 接口5：模型版本发布（将生成的最大模型版本号存入到数据库中）
### 接口6：删除最大版本号的模型（新训练的模型效果差，没必要进行保存操作）