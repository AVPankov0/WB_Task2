from catboost import CatBoostClassifier
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from conf import config
import preprocess.preprocess as preprocess
import preprocess.feature_importance as feature_importance
from sklearn.metrics import precision_score, recall_score, f1_score

if config().mode == 'train':
    logging.basicConfig(level=logging.INFO, filename="logs/logging.log",filemode="w", format="%(asctime)s %(levelname)s %(message)s")
    classifier = CatBoostClassifier(depth = config().catboost_params['depth'], iterations = config().catboost_params['iterations'], 
                                    l2_leaf_reg = config().catboost_params['l2_leaf_reg'], 
                                    loss_function = config().catboost_params['loss_function'], random_state=42)
    logging.info("Preprocessing data")
    df=preprocess.main()
    feature_importance.process(df.drop(['id1', 'id2', 'id3', 'text', 'label'], axis=1), df.label)
    X = df.drop(['id1', 'id2', 'id3', 'text', 'label'], axis=1)[pickle.load(open('../data/feat_list.sav', 'rb'))]
    y = df.label
    del df
    logging.info("Preprocessing complete")
    logging.info("Training")
    classifier.fit(X, y)
    logging.info("Training complete")
    pickle.dump(classifier, open('../models/classifier.sav', 'wb'))
elif config().mode == 'work':
    classifier = pickle.load(open('../models/classifier.sav', 'rb'))
    classifier.set_probability_threshold(config().catboost_params['threshold'])
    logging.info('Preprocessing data')
    df = preprocess.main()
    X = df.drop(['id1', 'id2', 'id3', 'text', 'label'], axis=1)[pickle.load(open('../data/feat_list.sav', 'rb'))]
    y = df.label
    del df
    logging.info('Preprocessing complete')
    y_pred = classifier.predict(X)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
else:
    logging.error("Unkown mode. Please use 'work' or 'train' only.")
    