# AI project - BERT大纲

## 模型的架构

pre-train模型架构 + softmax三分类器

![1592634344897](\1592634344897.png)

[Source](http://jalammar.github.io/illustrated-bert/)

![1592635033405](\1592635033405.png)

[Source](http://jalammar.github.io/illustrated-bert/)

BERTBASE ：L=12,H=768,A=12（总参数为110M。和GPT一样）

模型的细节：

[模型](https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1)

计算细节：

本地GPU： GTX 1070Ti



## 模型参数的设计

train/test划分： 

1. 抽样比例：80% / 20%
2. 是否平衡样本量：是
3. 是否对原样本进行等样本量subsample： 是(i.e. shuffle样本)

模型参数：

MAX_SEQ_LENGTH = 128

超参的设置来源于这里 [notebook](https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb) ，这些超参被google证实适用于广泛的下游任务

BATCH_SIZE = 32

LEARNING_RATE = 2e-5

NUM_TRAIN_EPOCHS = 3

(Warmup is a period of time where hte learning rate  is small and gradually increases--usually helps training.)

WARMUP_PROPORTION = 0.1

SAVE_CHECKPOINTS_STEPS = 500

SAVE_SUMMARY_STEPS = 100



## 模型的评价

假设样本为$(X_i,Y_i)$,其中$Y_i \in \{0,1,2\}$,预测的样本label为$\hat{Y_i}$

$accuracy = \frac{\sum_i I(Y_i = \hat{Y_i})}{\#TotaL\_Observations}$

$recall\_nClassifier = \frac{1}{\#Labels}\sum_{j\in\{0,1,2\}}recall(j)$

$precision\_nClassifier = \frac{1}{\#Labels}\sum_{j\in\{0,1,2\}}precision(j)$

其中 $recall(j), precision(j)$为将三分类问题变为二分类问题（label = j或label $\neq$ j）时对应的recall和precision



## 模型的结果

test data set summary

![1592636060393](\1592636060393.png)

从结果认为，test模型和train模型的loss差距不大，因此在这个模型的架构下可以认为模型基本处于一个fine-fit的状态。



## 源码

```AI_bert_for_final.py```

```python
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 2020

@credit to google's bert model
@author: robert
"""
#==============================set data set================================
DATA_SET = "sample_19252.json" #FIXME: change this to biggest data set!
DATA_COLUMN = 'newsSummary' #FIXME: can use 'newsTitle'
LABEL_COLUMN = 'emotionIndicator'
EXPORT_PATH = 'testing'

#==============================Import library ==============================
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

#==============================set parameter ==================================
label_list = [0,1,2]
TEST_SIZE = 0.2
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1"

MAX_SEQ_LENGTH = 128
# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3
# Warmup is a period of time where hte learning rate 
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100


#=============================main functions below ================================
# function to reconstruct json to get data frame
def reconstruct_json_to_dataframe(file):
  data = []
  for obs in range(len(file)):
    aRecord = file[obs]
    emotionIndicator = aRecord.get('emotionInfos')[0].get('emotionIndicator')
    newsId = aRecord.get('newsId')
    newsSummary = aRecord.get('newsSummary')
    newsTitle = aRecord.get('newsTitle')
    data.append([newsId, newsTitle, newsSummary, emotionIndicator])

  frame = pd.DataFrame(data, index = range(len(file)),columns = ['newsId','newsTitle','newsSummary','emotionIndicator'])

  return frame


# construct data frame for train vs test
def construct_dataFrame(featureArray, labelArray, columns):
  df = pd.DataFrame()
  df[columns[0]] = featureArray
  df[columns[1]] = labelArray
  return df


# use google's pre-trained model to get tokenizer
def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)


# create the model
def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,num_labels):
  """Creates a classification model."""
  bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True)
  bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
  bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

  # Use "pooled_output" for classification tasks on an entire sentence.
  # Use "sequence_outputs" for token-level output.
  output_layer = bert_outputs["pooled_output"]

  hidden_size = output_layer.shape[-1].value

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())


  with tf.variable_scope("loss"):
    # Dropout helps prevent overfitting
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
    # TRAIN and EVAL
    if not is_predicting:

      (loss, predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      train_op = bert.optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics. 
      def metric_fn(label_ids, predicted_labels, num_labels):
        accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
        recall_n = [0] * num_labels
        precision_n = [0] * num_labels
        update_op_rec_n = [[]] * num_labels
        update_op_pre_n = [[]] * num_labels
        for k in range(num_labels):
            recall_n[k], update_op_rec_n[k] = tf.metrics.recall(
                labels=tf.equal(label_ids, k),
                predictions=tf.equal(predicted_labels, k)
            )    
            precision_n[k], update_op_pre_n[k] = tf.metrics.precision(
                labels=tf.equal(label_ids, k),
                predictions=tf.equal(predicted_labels, k)
            )    
        recall_value = sum(recall_n) * 1.0 / num_labels
        precision_value = sum(precision_n) * 1.0 / num_labels
        update_op_rec = sum(update_op_rec_n) * 1.0 / num_labels
        update_op_pre = sum(update_op_pre_n) * 1.0 / num_labels
        recall = (recall_value, update_op_rec)
        precision = (precision_value, update_op_pre)
        return {
            "eval_accuracy": accuracy,
            "recall": recall,
            "precision": precision,
        }

      eval_metrics = metric_fn(label_ids, predicted_labels, num_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          train_op=train_op)
      else:
          return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:
      (predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn

# used to do prediction
def predict_emotion(dataList):
  newDf = pd.DataFrame()
  newDf[DATA_COLUMN] = dataList
  newDf[LABEL_COLUMN] = [1]*len(dataList)

  input_examples = newDf.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping
                                                                          text_a = x[DATA_COLUMN], 
                                                                          text_b = None, 
                                                                          label = x[LABEL_COLUMN]), axis = 1) # here, "" is just a dummy label
  input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
  predict_input_fn = run_classifier.input_fn_builder(features=input_features, 
                                                    seq_length=MAX_SEQ_LENGTH, 
                                                    is_training=False, 
                                                    drop_remainder=False)
  predictions = estimator.predict(predict_input_fn)

  for p in predictions:
    print("probabilities: " + str(p["probabilities"]) + "label is: " + str(label_list[p['labels']]) + "\n")

def serving_input_fn():
    label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
    input_ids = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH], name='segment_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        {
        'label_ids': label_ids,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids
        }
    )()
    return input_fn

#===============================run ==============================================
if __name__ == '__main__':
    with open(DATA_SET, 'r', encoding='utf-8') as file:
        papers = []
        for line in file.readlines():
            dic = json.loads(line)
            papers.append(dic)

    # get dataframe from json
    rawData = reconstruct_json_to_dataframe(papers)

    # check if GPU is online
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    # prepare a folder to store results
    OUTPUT_DIR = 'BERT_OUT_DIR'
    # Whether or not to clear/delete the directory and create a new one
    DO_DELETE = True

    if DO_DELETE:
        try:
            tf.gfile.DeleteRecursively(OUTPUT_DIR)
        except:
            # Doesn't matter if the directory didn't exist
            pass
    tf.gfile.MakeDirs(OUTPUT_DIR)
    print('***** Model output directory: {} *****'.format(OUTPUT_DIR))

    # split the data set to get train and test
    X_train, X_test, y_train, y_test = train_test_split(rawData[DATA_COLUMN],
                                                    rawData[LABEL_COLUMN], 
                                                    test_size = TEST_SIZE, 
                                                    shuffle = True,
                                                    stratify = rawData[LABEL_COLUMN])

    train = construct_dataFrame(X_train, y_train, [DATA_COLUMN, LABEL_COLUMN])
    test = construct_dataFrame(X_test, y_test, [DATA_COLUMN, LABEL_COLUMN])

    # Use the InputExample class from BERT's run_classifier code to create examples from the data
    train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping
                                                                                text_a = x[DATA_COLUMN], 
                                                                                text_b = None, 
                                                                                label = x[LABEL_COLUMN]), axis = 1)

    test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                                text_a = x[DATA_COLUMN], 
                                                                                text_b = None, 
                                                                                label = x[LABEL_COLUMN]), axis = 1)

    # init tokenizer
    tokenizer = create_tokenizer_from_hub_module()

    # Convert our train and test features to InputFeatures that BERT understands.
    train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
    test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
    
    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
    print("\n number of train steps: "+str(num_train_steps))

    # Specify output directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    # build bert model
    model_fn = model_fn_builder(
                                num_labels=len(label_list),
                                learning_rate=LEARNING_RATE,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
                                        model_fn=model_fn,
                                        config=run_config,
                                        params={"batch_size": BATCH_SIZE})

    # Create an input function for training.
    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)

    # start training.
    print('Beginning Training!')
    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("Training took time ", datetime.now() - current_time)

    # test and evaluate
    test_input_fn = run_classifier.input_fn_builder(
                                                    features=test_features,
                                                    seq_length=MAX_SEQ_LENGTH,
                                                    is_training=False,
                                                    drop_remainder=False)
    estimator.evaluate(input_fn=test_input_fn, steps=None)

    # run some test data
    print(test[DATA_COLUMN].iloc[:10].values)
    predict_emotion(test[DATA_COLUMN].iloc[:10].values)

    # save model to pb file in testing folder
    estimator._export_to_tpu = False  # NEVER CHANGE THIS LINE
    estimator.export_saved_model(export_dir_base=EXPORT_PATH,serving_input_receiver_fn=serving_input_fn)
```



```load_model_and_out_csv.py```

```python
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 2020

@credit to google's bert model
@author: robert
"""
#==============================Import library ==============================
import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from tensorflow.contrib import predictor
from datetime import datetime

#==============================set parameter & dataset==================================
label_list = [0,1,2]
MAX_SEQ_LENGTH = 128
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1"
USE_DATA_COLUMN = 'content' #FIXME: can change to 'title'
EXPORT_PATH = 'testing'

newsDataFileName = 'sinaCompany_'+datetime.today().strftime("%Y%m%d")+'.csv'
toWriteFileName = 'news_' + datetime.today().strftime("%Y%m%d") + '.csv'

#=============================main functions below ================================
def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)


def predict(sentences, predict_fn):
    labels = label_list
    input_examples = [
        run_classifier.InputExample(
            guid="",
            text_a = x,
            text_b = None,
            label = 0
        ) for x in sentences] # here, "" is just a dummy label
    input_features = run_classifier.convert_examples_to_features(
        input_examples, labels, MAX_SEQ_LENGTH, tokenizer
    )

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in input_features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)
    pred_dict = {
        'input_ids': all_input_ids,
        'input_mask': all_input_mask,
        'segment_ids': all_segment_ids,
        'label_ids': all_label_ids
    }

    predictions = predict_fn(pred_dict)
    return [
        (sentence, prediction, label)
        for sentence, prediction, label in zip(pred_sentences, predictions['probabilities'], predictions['labels'])
    ]

#===============================run ==============================================
if __name__ == '__main__':
    # get news data today
    newsDataSet = pd.read_csv(newsDataFileName)

    # create tokenizer and prection function
    outputFileName = os.listdir(EXPORT_PATH)[-1]
    tokenizer = create_tokenizer_from_hub_module()
    predict_fn = predictor.from_saved_model(EXPORT_PATH + '/' + outputFileName)
    
    # do prediction
    pred_sentences = newsDataSet[USE_DATA_COLUMN].values
    predictions = predict(pred_sentences, predict_fn)
    classification = [context[2] for context in predictions]

    # prepare output data
    outData = pd.DataFrame(columns = ['title'])
    outData['title'] = sinaData['title'].values
    outData['url'] = sinaData['link'].values
    outData['classification'] = classification

    # save to csv
    outData.to_csv(toWriteFileName)
```

