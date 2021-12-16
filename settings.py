# -*- ecoding: utf-8 -*-
# @ModuleName: settings
# @Function: 写一些全局变量，用于数据库的相应配置
# @Author: long
# @Time: 2021/12/14 10:39
# *****************conding***************

import os
MYSQL_SETTING = {
    'default': {
        # 'hosts': '192.168.37.22:9200'
        'hosts': os.environ.get('MYSQL_HOST', '202.107.190.8'),
        'port':os.environ.get('MYSQL_PORT', 10811),
        "username":os.environ.get('USER', 'root'),
        'password':os.environ.get("PASSWORD","LHCZroot123456"),
        'dbname':os.environ.get("DBNAME", "knowledge_graph_project")
    },
}
