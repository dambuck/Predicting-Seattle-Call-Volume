
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.model_selection import TimeSeriesSplit


class metrics(object):
    '''
    class containing metrics for evaluation
    '''
    def MAE(self,X,Y):
        return np.abs(X-Y).mean()
    
class data_loader(object):
    '''
    Load prepared train and test data
    
    train_features: column names of train.csv used for training
    target_features: ""
    path: provide path where train.csv and test.csv are locaated
    '''
    def load_train_test(self,train_features=['year', 'month', 'day', 'hour', 'week',
                        'PRCP', 'SNOW', 'SNWD','TMAX', 'TMIN'],
                        target_features=["volume"],
                        path=""):
        
        if len(path)==0:
            parent=os.path.dirname(os.getcwd())
            path=os.path.join(parent,"input")
        
        train=pd.read_csv(os.path.join(path, "train.csv"))
        test=pd.read_csv(os.path.join(path, "test.csv"))
        
        if len(target_features)==1:
            target_features=target_features[0]
        if len(train_features)==1:
            train_features=train_features[0]
        
        train_x=train[train_features]
        train_y=train[target_features]

        test_x=test[train_features]
        test_y=test[target_features]
        
        return train_x, train_y, test_x, test_y

    
class RF_Model(metrics, data_loader):
    
    def __init__(self,params={}):
        '''
        initialize RF regressor
        
        Params: dictionary with RandomForest parameters
        '''
        self.parent=os.path.dirname(os.getcwd())
        
        if len(params)==0:
            self.model = RandomForestRegressor()
        else:
            try:
                self.model = RandomForestRegressor(**params)
                print("initialized with params")
            except:
                print("non adequat parameters !")
        
    def train(self, X,Y):
        '''
        fit model to Data
        
        X: training data
        Y: training targets
        
        '''
        self.model.fit(X,Y)
    
    def predict_(self,X):
        '''
        make predictions
        
        X: Data array
        
        returns predictions
        '''
        self.forecast=self.model.predict(X)
        return self.forecast

    
    def evaluate(self,X,Y,metric="MAE"):
        '''
        compute metric
        '''
        if metric=="MAE":
            try:
                return self.MAE(self.forecast,Y)
            except:
                return self.MAE(self.model.predict(X),Y)
        else:
            print("metric not available")

    def time_cv_index(self,X,n_splits):
        '''
        compute indices for time series cross validation
        
        X: training data
        n_splits: number of splits for cross validation
        '''
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.CV=[]
        for train_index, test_index in self.tscv.split(X):
            self.CV.append((train_index, test_index))
        
    def random_search(self,X,Y,n_splits,iters,search_params,seed=0):
        '''
        perform random hypterparameter search
        
        X: training data
        Y: training targets
        n_splits: number of splits for cross validation
        iters: sampling iterations
        search_params: dictionary with parameters distributions
        seed: for reproducebility
        '''
        self.time_cv_index(X,n_splits)
        
        self.rs = RandomizedSearchCV(estimator=self.model, param_distributions=search_params,
                                     cv=self.CV,scoring="neg_mean_absolute_error", 
                         random_state=seed, refit="neg_mean_absolute_error", n_iter=iters)
        
        self.rs_results = self.rs.fit(X, Y)
        self.model=self.rs.best_estimator_
        
    def grid_search(self,X,Y,n_splits, search_params):
        '''
        perform grid search
        
        X: training data
        Y: training targets
        n_splits: number of splits for cross validation
        search_params: dictionary with parameters 
        
        Might take long!
        
        '''
        self.time_cv_index(X,n_splits)
        self.gs = GridSearchCV(estimator=self.model, param_grid=search_params,
                               cv=self.CV,scoring="neg_mean_absolute_error", refit="neg_mean_absolute_error")
        self.gs_results=self.gs.fit(X,Y)
        self.model=self.gs.best_estimator_
        
    def Hyper_opt(self,X,Y,n_splits,iters,search_params,perc=0.1,seed=0):
        '''
        perform random search and subsequent grid search in best_params neighbourhood
        
        
        '''
        
        self.random_search(X,Y,n_splits,iters,search_params,seed)
        self.neighbour_grid={}
        
        for key in self.rs_results.best_params_.keys():
            self.neighbour_grid[key]=[int(self.rs_results.best_params_[key]*(1-perc)),
                                      self.rs_results.best_params_[key],
                            int(np.ceil(self.rs_results.best_params_[key]*(1+perc)))]
            
        self.grid_search(X,Y,n_splits,self.neighbour_grid)
        
    def save_model(self,path=""):
        '''
        saves model with pickle
        '''
        
        if len(path)==0:
            path=os.path.join(self.parent,"models")
            path=os.path.join(path,"RF_model.sav")
        
        pickle.dump(self.model, open(path, 'wb'))

        print("saved model to location: ", path)
        
    def load_model(self,path=""):
        
        if len(path)==0:
            path=os.path.join(self.parent,"models")
            path=os.path.join(path,"best_RF_model.sav")
        
        self.model = pickle.load(open(path, 'rb'))
        
