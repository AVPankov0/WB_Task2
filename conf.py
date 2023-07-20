#best parameters for CatBoost model according to GridSearch
cbp =  {'depth': 3,  'iterations': 40, 'l2_leaf_reg': 0.00020691380811147902,
                             'loss_function': 'Logloss', 'threshold': 0.428571}
# Behaviour mode: 'train' for fitting model on new data, 'work' for predicting labels for unseen data
md = 'train'
class config:
    def __init__(self, catboost_params = cbp, mode = md):
        self.catboost_params =  catboost_params
        self.mode = mode