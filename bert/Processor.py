
import tensorflow as tf
import csv
import os
import sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
print(BASE)

import bert_base.modeling
import bert_base.tokenization
import tensorflow as tf
from Processor import *
from bert_base.run_classifier import *

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        raise NotImplementedError()




class NewsSentimentProcessor(DataProcessor):
    """2019.06.30-hjy-进行新闻的情感分析，负面or非负面"""

    def __init__(self):
        self.labels = ['0', '1'] # 区分新闻情感

    def get_train_examples(self, data_dir):
        '''
        :param data_dir:/home/hjy/nlp_dataset/xiniu_news_sentiment_nlp_dataset/
        :return:
        '''
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_data.csv")), "train") #

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_data.csv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_data.csv")), "test")

    def get_labels(self):
        return self.labels

    
    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        examples = []

        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if(i==0):
                continue
            # print(i)
            try:
                text_a = tokenization.convert_to_unicode(line[1])
                label = tokenization.convert_to_unicode(line[0])
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            except:
                print('error example:',i)

        return examples
