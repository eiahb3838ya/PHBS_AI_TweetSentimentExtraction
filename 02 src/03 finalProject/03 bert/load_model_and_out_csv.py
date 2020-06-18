# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 2020

@credit to google's bert model
@author:Robert
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

