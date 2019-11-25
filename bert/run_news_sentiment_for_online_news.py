#! usr/bin/env python3
# -*- coding:utf-8 -*-

# Copyright 2018 The Google AI Language Team Authors.
# BASED ON Google_BERT.
# @Author: huajingyun
# @Date: 2019-08-14
# @Last Modified by: huajingyun
# @Last Modified time: 2019-08-14
# @Reference：https://github.com/google-research/bert
# @Description: ﻿online news 情感分析结果入库

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 读取父级目录路径
import sys
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
print(BASE)

import json
import os
import bert_base.modeling as modeling
import bert_base.tokenization as tokenization
from bert_base.run_classifier import *
import pandas as pd

from tqdm import tqdm
import logging

# log:
logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)
import pymysql
import tensorflow as tf
BERT_BASE_DIR='/home/hjy/corpora/chinese_L-12_H-768_A-12/'

## Required parameters
data_dir = '/home/hjy/nlp_dataset/xiniu_news_sentiment_nlp_dataset/'
bert_config_file = BERT_BASE_DIR+'bert_config.json'
task_name = 'newssentiment'
vocab_file = BERT_BASE_DIR+'vocab.txt'
output_dir = '/home/hjy/sv/nlp/nlp_tasks_based_on_BERT/news_sentiment/output'
init_checkpoint = '/home/hjy/sv/nlp/bert-master/newssentiment_output_wwm/model.ckpt-4000'

# Other parameters
do_lower_case = True
max_seq_length = 128
do_predict = True
train_batch_size = 32
eval_batch_size = 8
predict_batch_size = 8
learning_rate = 5e-5
num_train_epochs = 3.0
warmup_proportion = 0.1
save_checkpoints_steps = 1000
iterations_per_loop = 1000
use_tpu = False
master=None
num_tpu_cores=8

class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls):
    # *** sql读取news文本 ***
    # config
    localhost = '172.26.128.119'
    username = 'dev'
    password = 'Chuangxin_2018svbdev'
    dbname = 'spider_news'
    port = 3306

    # sql:
    sql_cmd = "select id,title,summary from news_online"  # limit 10

    # 用DBAPI构建数据库链接engine
    con = pymysql.connect(host=localhost, user=username, password=password, port=port, database=dbname,
                          charset='utf8', use_unicode=True)
    df = pd.read_sql(sql_cmd, con)  # bad news
    logger.info("connect sql successfully..")
    print(df.shape)
    df.dropna(subset=['title', 'id'], inplace=True)
    print(df.shape)
    list(df['title'])
    # *** sql读取news文本 END ***
    lines=list(df['title'])
    return lines

def _read_ids():
    # *** sql读取news文本 ***
    # config
    localhost = '172.26.128.119'
    username = 'dev'
    password = 'Chuangxin_2018svbdev'
    dbname = 'spider_news'
    port = 3306

    # sql:
    sql_cmd = "select id,title,summary from news_online"  # limit 10

    # 用DBAPI构建数据库链接engine
    con = pymysql.connect(host=localhost, user=username, password=password, port=port, database=dbname,
                          charset='utf8', use_unicode=True)
    df = pd.read_sql(sql_cmd, con)  # bad news
    logger.info("connect sql successfully..")
    print(df.shape)
    df.dropna(subset=['title', 'id'], inplace=True)
    print(df.shape)
    #
    ids=list(df['id'])
    return ids


class NewsSentimentProcessor(DataProcessor):
    """2019.06.30-hjy-进行新闻的情感分析，负面or非负面"""

    def __init__(self):
        self.labels = ['负面', '非负面'] # 区分新闻情感

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(), "test")

    def get_labels(self):
        return self.labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        examples = []

        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # print(i)

            text_a = tokenization.convert_to_unicode(line)
            label = '非负面'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples

def to_sql_com_name(input_text, id):
    '''
    将结果sql数据库
    依照id完成NLPTags字段填充，写入sentiment的json
    '''
    # *** sql读取news文本 ***
    # config
    localhost = '172.26.128.119'
    username = 'dev'
    password = 'Chuangxin_2018svbdev'
    dbname = 'spider_news'
    port = 3306
    # 用DBAPI构建数据库链接engine
    con = pymysql.connect(host=localhost, user=username, password=password, port=port, database=dbname,
                          charset='utf8', use_unicode=True)

    with con.cursor() as cursor:
        update_sql = "UPDATE `news_online` set `NLPsentiment` = '" + input_text  + "' where `id` = " + str(id)
        try:
            cursor.execute(update_sql)
        except Exception as e:
            print(e)
            print(update_sql)
        con.commit()

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "newssentiment": NewsSentimentProcessor,
  }

  tokenization.validate_case_matches_checkpoint(do_lower_case,
                                                init_checkpoint)


  bert_config = modeling.BertConfig.from_json_file(bert_config_file)

  if max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(output_dir)

  task_name1 = task_name.lower()

  if task_name1 not in processors:
    raise ValueError("Task not found: %s" % (task_name1))

  processor = processors[task_name1]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

  tpu_cluster_resolver = None


  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=master,
      model_dir=output_dir,
      save_checkpoints_steps=save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=iterations_per_loop,
          num_shards=num_tpu_cores,
          per_host_input_for_training=is_per_host))

  num_train_steps = None
  num_warmup_steps = None


  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=init_checkpoint,
      learning_rate=learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=use_tpu,
      use_one_hot_embeddings=use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=train_batch_size,
      eval_batch_size=eval_batch_size,
      predict_batch_size=predict_batch_size)

  if do_predict:
    print('do_predict')
    predict_examples = processor.get_test_examples(data_dir)
    num_actual_predict_examples = len(predict_examples)
    print('**num_actual_predict_examples',num_actual_predict_examples)

    predict_file = os.path.join(output_dir, "predict.tf_record")

    file_based_convert_examples_to_features(predict_examples, label_list,
                                            max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", predict_batch_size)

    predict_drop_remainder = True if use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)
    print("***** Predict results *****")
    result = estimator.predict(input_fn=predict_input_fn)
    ids=_read_ids()

    for (id, prediction) in tqdm(zip(ids,result)):
        output={}
        # print(i+1,prediction["probabilities"][0])
        if prediction["probabilities"][0]>0.15:
            output['sentiment']='负面'
        else:
            output['sentiment']='非负面'
        output['sentiment_prob']= str(prediction["probabilities"][0])

        to_sql_com_name(json.dumps(output,ensure_ascii=False), id)
        # print("\t".join(str(class_probability) for class_probability in probabilities) + "\n")
        print('complete...')

if __name__ == "__main__":
    tf.app.run()
