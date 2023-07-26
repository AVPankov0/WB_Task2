from catboost import CatBoostClassifier
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from conf import Config
import preprocess.preprocess as preprocess
import preprocess.feature_importance as feature_importance
from sklearn.metrics import precision_score, recall_score, f1_score

if Config().mode == 'train':
    logging.basicConfig(level=logging.INFO, filename="./logs/logging.log",filemode="w", format="%(asctime)s %(levelname)s %(message)s")
    classifier = CatBoostClassifier(depth = Config().catboost_params['depth'], iterations = Config().catboost_params['iterations'], 
                                    l2_leaf_reg = Config().catboost_params['l2_leaf_reg'], 
                                    loss_function = Config().catboost_params['loss_function'], random_state=42)
    logging.info("Preprocessing data")
    df=preprocess.preprocess(Config().data_path)
    feature_importance.process(df.drop(['id1', 'id2', 'id3', 'text', 'label'], axis=1), df.label, Config().feature_path)
    X = df.drop(['id1', 'id2', 'id3', 'text', 'label'], axis=1)[pickle.load(open(Config().feature_path, 'rb'))]
    y = df.label
    del df
    logging.info("Preprocessing complete")
    logging.info("Training")
    classifier.fit(X, y)
    logging.info("Training complete")
    pickle.dump(classifier, open(Config().classifier_path, 'wb'))
    
elif Config().mode == 'work':
    logging.basicConfig(level=logging.INFO, filename="./logs/logging.log",filemode="w", format="%(asctime)s %(levelname)s %(message)s")
    classifier = pickle.load(open(Config().classifier_path, 'rb'))
    classifier.set_probability_threshold(Config().catboost_params['threshold'])
    logging.info('Preprocessing data')
    df = preprocess.preprocess(Config().data_path)
    X = df.drop(['id1', 'id2', 'id3', 'text', 'label'], axis=1)[pickle.load(open(Config().feature_path, 'rb'))]
    y = df.label
    del df
    logging.info('Preprocessing complete')
    y_pred = classifier.predict(X)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    pd.Series(y_pred).to_csv('./data/predicted_labels.csv')
    logging.info('Precision score: '+str(precision))
    logging.info('Recall score: '+str(recall))
    logging.info('F1 score: '+str(f1))
else:
    logging.error("Unkown mode. Please use 'work' or 'train' only.")
    