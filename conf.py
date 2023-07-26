#best parameters for CatBoost model according to GridSearch
cbp =  {'depth': 3,  'iterations': 40, 'l2_leaf_reg': 0.00020691380811147902,
                             'loss_function': 'Logloss', 'threshold': 0.428571}

#path to data, model and feature list
dat_path = './data/wb_school_task_2.csv.gzip'
cf_path = './models/classifier.sav'
feat_path = './data/feat_list.sav'

# Behaviour mode: 'train' for fitting model on new data, 'work' for predicting labels for unseen data
md = 'work'

class Config:
    def __init__(self, catboost_params = cbp, mode = md, data_path = dat_path, classifier_path = cf_path, feature_path = feat_path):
        self.catboost_params =  catboost_params
        self.mode = mode
        self.data_path = data_path
        self.classifier_path = classifier_path
        self.feature_path = feature_path