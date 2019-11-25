#! usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: huajingyun
# @Date:   2019-05-04 11:01:01
# @Last Modified by:   huajingyun
# @Last Modified time: 2019-05-08 12:01:01
# @Description:  构建数据集
#  读取sql库中的新闻语料，存储为txt格式
#  每行一个样本，每行格式：分类标签\t文本\n

# 载入接下来分析用的库
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
import pymysql
import logging
from config import *
import time  # 引入time模块
 
start = time.time()
# sys.getsizeof(df2)/1024**3
logging.basicConfig(level=logging.INFO, filename='./log/sentiment.log',filemode='a',format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
# log:
logger = logging.getLogger(__name__)

# config
FILENAME  = './xiniu_news_sentiment.txt'

# sql:
sql_cmd = "select title,brief,features,sentiment,article_type,industry_type from xiniu_news" # limit 10
logger.info(sql_cmd)
# 用DBAPI构建数据库链接engine
con = pymysql.connect(host=LOCALHOST, user=USERNAME, password=PASSWORD, port=PORT, database=DBNAME_XINIU, charset='utf8', use_unicode=True)
df = pd.read_sql(sql_cmd, con) # bad news
logger.info("connect sql successfully..")
print(df.shape)
df.dropna(subset=['title','brief','features','sentiment','article_type','industry_type'],inplace=True)
print(df.shape)

# shuffle
df = shuffle(df)
df.reset_index(drop=True,inplace=True)

# 存成txt
with open(FILENAME,'w') as f:
    for i in tqdm(range(len(df))):
        ss = "".join(df.iloc[i]['title'].split()+df.iloc[i]['brief'].split())
        f.write(df.iloc[i]['sentiment']+'\t'+ss+'\n')
f.close()
end = time.time()
duration=end-start
logger.info("toatal run time is "+ str(duration))
logger.info("save {} completely\n".format(FILENAME))





