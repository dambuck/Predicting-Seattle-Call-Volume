
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from model import RF_Model

# input directory, contains training and testing data
parent=os.path.dirname(os.getcwd())
input_loc=os.path.join(parent,"input")
model_path=os.path.join(parent,"models")

train_features=['year', 'month', 'day', 'hour', 'week','PRCP', 'SNOW', 'SNWD','TMAX', 'TMIN']
target_features=["volume"]
time_cv_split=5
random_iter=2
perc=0.20

param_grid={
    'n_estimators': [5],
    'min_samples_split': [2],
    'min_samples_leaf': [4 ],
    'max_features': [0.2],
    'max_depth': [10]
}



# baseline
print("establish baseline model")
model=RF_Model(params={"n_estimators":100, "random_state":0})

train_x,train_y, test_x, test_y=model.load_train_test(train_features=train_features,
                                                     target_features=target_features,path=input_loc)
        


model.train(train_x, train_y)
baseline_score=model.evaluate(test_x, test_y)
print("baseline MAE: " ,baseline_score)
model.save_model(path=os.path.join(model_path,"baseline_RF_model.sav"))

print("perform hyperparameter optimization")
model.Hyper_opt(X=train_x,Y=train_y,n_splits=2,iters=1,search_params=param_grid,perc=0.1,seed=0)
score_after_opt=model.evaluate(test_x, test_y)
print("optimized MAE: ",score_after_opt)
print("MAE reduction: {:0.4f}%. ".format(100*(score_after_opt-baseline_score)/baseline_score))
model.save_model(path=os.path.join(model_path,"best_RF_model.sav"))
