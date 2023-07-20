import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, f1_score
import pickle
import logging

def feature_select(X_t, y_t, X_p, y_p, feat_list, num_feat=1, ignored_feat=[], estimator=LogisticRegression(max_iter=10000)):
    if ignored_feat:
        for feat in ignored_feat:
            feat_list.drop(feat_list.loc[feat_list == feat].index, inplace=True)
    X_train = X_t[feat_list[0:num_feat]]
    y_train = y_t
    X_test = X_p[feat_list[0:num_feat]]
    y_test = y_p
    estimator = estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_p)
    return {'feat list': feat_list[0:num_feat], 'precision': precision_score(y_p, y_pred), 'f1': f1_score(y_p, y_pred)}
logreg = LogisticRegression(max_iter = 10000)
def process(X, y):
    logging.basicConfig(level=logging.INFO, filename="\log\logging.log",filemode="w", format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Selecting important features")
    logreg.fit(X,y)
    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(logreg.coef_[0])})
    feat_imp.sort_values(by='Importance', inplace=True, ascending=False)
    feat_imp.reset_index(inplace=True)
    a=[i for i in range(0, feat_imp.loc[feat_imp['Feature'] == 'rand_f'].index[0])]
    b=[i for i in range(feat_imp.loc[feat_imp['Feature'] == 'rand_f'].index[0], 236)]
    not_imp_feat = feat_imp.drop(index = a)
    imp_feat = pd.Series(data= feat_imp.drop(index = b).Feature)
    X_t, X_p, y_t, y_p =  train_test_split(X, y, test_size=0.2, random_state=42)
    X_t = X_t.drop(columns = [i for i in not_imp_feat['Feature']])[imp_feat]
    X_p = X_p.drop(columns = [i for i in not_imp_feat['Feature']])[imp_feat]
    ignored_feat = []
    last_precision = 0
    last_f1=0
    last_list = []
    tol = 0.005
    num_feat=1
    for i in range(1, len(imp_feat)):
        feat_list=imp_feat.copy()
        out = feature_select(X_t=X_t, y_t=y_t, X_p=X_p, y_p=y_p, num_feat=num_feat, ignored_feat=ignored_feat, 
                             feat_list=feat_list, estimator = CatBoostClassifier(verbose=False, random_state=42))
        if (out['precision']-last_precision <= tol) and (out['f1']-last_f1 <= tol):
            ignored_feat.append(out['feat list'].iloc[-1]) 
            last_precision=out['precision']
            last_f1=out['f1']
        else:
            last_precision=out['precision']
            last_f1=out['f1']
            num_feat+=1
            final_feat = out['feat list']
    del a, b, X_t, X_p, y_y, y_p
    logging.info("Done")
    pickle.dump(final_feat,  open('feat_list.sav', 'wb'))