
import tensorflow as tf
import csv
import os


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


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

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        if len(line) == 0: continue
        lines.append(line)
      return lines


class GLUEProcessor(DataProcessor):
  def __init__(self):
    self.train_file = "train.tsv"
    self.dev_file = "dev.tsv"
    self.test_file = "test.tsv"
    self.label_column = None
    self.text_a_column = None
    self.text_b_column = None
    self.contains_header = True
    self.test_text_a_column = None
    self.test_text_b_column = None
    self.test_contains_header = True

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train_data.csv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_data.csv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    if self.test_text_a_column is None:
      self.test_text_a_column = self.text_a_column
    if self.test_text_b_column is None:
      self.test_text_b_column = self.text_b_column

    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_data.csv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0 and self.contains_header and set_type != "test":
        continue
      if i == 0 and self.test_contains_header and set_type == "test":
        continue
      guid = "%s-%s" % (set_type, i)

      a_column = (self.text_a_column if set_type != "test" else
          self.test_text_a_column)
      b_column = (self.text_b_column if set_type != "test" else
          self.test_text_b_column)

      # there are some incomplete lines in QNLI
      if len(line) <= a_column:
        tf.logging.warning('Incomplete line, ignored.')
        continue
      text_a = line[a_column]

      if b_column is not None:
        if len(line) <= b_column:
          tf.logging.warning('Incomplete line, ignored.')
          continue
        text_b = line[b_column]
      else:
        text_b = None

      if set_type == "test":
        label = self.get_labels()[0]
      else:
        if len(line) <= self.label_column:
          tf.logging.warning('Incomplete line, ignored.')
          continue
        label = line[self.label_column]
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class Yelp5Processor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_examples(os.path.join(data_dir, "train.csv"))

  def get_dev_examples(self, data_dir):
    return self._create_examples(os.path.join(data_dir, "test.csv"))

  def get_labels(self):
    """See base class."""
    return ["1", "2", "3", "4", "5"]

  def _create_examples(self, input_file):
    """Creates examples for the training and dev sets."""
    examples = []
    with tf.gfile.Open(input_file) as f:
      reader = csv.reader(f)
      for i, line in enumerate(reader):

        label = line[0]
        text_a = line[1].replace('""', '"').replace('\\"', '"')
        examples.append(
            InputExample(guid=str(i), text_a=text_a, text_b=None, label=label))
    return examples


class ImdbProcessor(DataProcessor):
  def get_labels(self):
    return ["neg", "pos"]

  def get_train_examples(self, data_dir):
    return self._create_examples(os.path.join(data_dir, "train"))

  def get_dev_examples(self, data_dir):
    return self._create_examples(os.path.join(data_dir, "test"))

  def _create_examples(self, data_dir):
    examples = []
    for label in ["neg", "pos"]:
      cur_dir = os.path.join(data_dir, label)
      for filename in tf.gfile.ListDirectory(cur_dir):
        if not filename.endswith("txt"): continue

        path = os.path.join(cur_dir, filename)
        with tf.gfile.Open(path) as f:
          text = f.read().strip().replace("<br />", " ")
        examples.append(InputExample(
            guid="unused_id", text_a=text, text_b=None, label=label))
    return examples


class MnliMatchedProcessor(GLUEProcessor):
  def __init__(self):
    super(MnliMatchedProcessor, self).__init__()
    self.dev_file = "dev_matched.tsv"
    self.test_file = "test_matched.tsv"
    self.label_column = -1
    self.text_a_column = 8
    self.text_b_column = 9

  def get_labels(self):
    return ["contradiction", "entailment", "neutral"]


class XnliProcessor(DataProcessor):
  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir, set_type="train"):
    """See base class."""
    train_file = os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language)
    lines = self._read_tsv(train_file)
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = line[0].replace(' ','')
      text_b = line[1].replace(' ','')
      label = line[2]
      if label == "contradictory":
        label = "contradiction"
      
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples    

  def get_devtest_examples(self, data_dir, set_type="dev"):
    """See base class."""
    devtest_file = os.path.join(data_dir, "xnli."+set_type+".tsv")
    tf.logging.info("using file %s" % devtest_file)
    lines = self._read_tsv(devtest_file)
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      language = line[0]
      if language != self.language:
        continue

      text_a = line[6].replace(' ','')
      text_b = line[7].replace(' ','')
      label = line[1]
      
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]



class CSCProcessor(DataProcessor):
  def get_labels(self):
    return ["0", "1","2"]

  def get_train_examples(self, data_dir):
    set_type = "train"
    input_file = os.path.join(data_dir, "train_data.csv")
    tf.logging.info("using file %s" % input_file)
    lines = self._read_tsv(input_file)
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)

      text_a = line[1]
      label = line[0]
    
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples
  
  def get_devtest_examples(self, data_dir, set_type="dev"):
    # input_file = os.path.join(data_dir, "dev_data.csv")
    input_file = os.path.join(data_dir, set_type+"_data.csv")
    tf.logging.info("using file %s" % input_file)
    lines = self._read_tsv(input_file)
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)

      text_a = line[1]
      label = line[0]
    
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

class CarForumProcessor(DataProcessor):
  def get_labels(self):
    return ["0", "1"]

  def get_train_examples(self, data_dir):
    set_type = "train"
    input_file = os.path.join(data_dir, "train_data.csv")
    tf.logging.info("using file %s" % input_file)
    lines = self._read_tsv(input_file)
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)

      text_a = line[1]
      label = line[0]
    
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples
  
  def get_devtest_examples(self, data_dir, set_type="dev"):
    # input_file = os.path.join(data_dir, "dev_data.csv")
    input_file = os.path.join(data_dir, set_type+"_data.csv")
    tf.logging.info("using file %s" % input_file)
    lines = self._read_tsv(input_file)
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)

      text_a = line[1]
      label = line[0]
    
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples



class MnliMismatchedProcessor(MnliMatchedProcessor):
  def __init__(self):
    super(MnliMismatchedProcessor, self).__init__()
    self.dev_file = "dev_mismatched.tsv"
    self.test_file = "test_mismatched.tsv"


class StsbProcessor(GLUEProcessor):
  def __init__(self):
    super(StsbProcessor, self).__init__()
    self.label_column = 9
    self.text_a_column = 7
    self.text_b_column = 8

  def get_labels(self):
    return [0.0]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0 and self.contains_header and set_type != "test":
        continue
      if i == 0 and self.test_contains_header and set_type == "test":
        continue
      guid = "%s-%s" % (set_type, i)

      a_column = (self.text_a_column if set_type != "test" else
          self.test_text_a_column)
      b_column = (self.text_b_column if set_type != "test" else
          self.test_text_b_column)

      # there are some incomplete lines in QNLI
      if len(line) <= a_column:
        tf.logging.warning('Incomplete line, ignored.')
        continue
      text_a = line[a_column]

      if b_column is not None:
        if len(line) <= b_column:
          tf.logging.warning('Incomplete line, ignored.')
          continue
        text_b = line[b_column]
      else:
        text_b = None

      if set_type == "test":
        label = self.get_labels()[0]
      else:
        if len(line) <= self.label_column:
          tf.logging.warning('Incomplete line, ignored.')
          continue
        label = float(line[self.label_column])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    return examples