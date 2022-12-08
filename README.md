# Predicting-Seattle-Call-Volume

The project is split into a config file, proprocessing, training and inference, according to the problem sheet. Jupyter notebooks are provided for writing the code via cell magic to .py files. I evaluated the models predictive power through MAE, as the absolute error directly reflect the deviation of predicted volume of calls from the true volume. Should be further improved to acount for type of prediction error. Obviously predicting a lower volume as the target volume is more problematic because the fire department will be short on stuff if following the prediction. Predicting a higher than target volume is less severe. Nevertheless this would lead to unnecesary additional spending because of additional stuff. However, I think that predictions should be taken as a lower bound for the call volume at all times, because of the prior reasoning. 
In the data analysis it became visible, that the call volume changed its general trend starting in 2020. One could see a decrease in the periodicity in the years starting in 2020 and and seemingly increase in fluctuation. This might be due to the pandemic. 


I provide the following folders:
SRC: The jupyter notebook for writing the files, aswell as all .py are here
model: Trained models are saved here via pickle
input: All model inputs are here. This include training and testing data
output: model predictions are saved here as csv

project_config.py: Use this to specify the paths for data and models, parameter grid for parameter tuning.

preprocessing.py: Load the raw Seattle call data and aditional weather data. Saves the processed data as train.csv and test.csv. Training data includes
all data up to the "last_year" variable in the config file. Default training data is from 2017 to 2021. 2022 is considered testing data.

model.py: Holds the model class. I used the sklear RandomforestRegressor. The class provides methods for fitting and predicting. It also includes a method for hyperparameter tuning. The class inherits from an extendable metric class. 

train_and_opt.py: Train a baseline model and do subsequent parameter tuning. The best model is refit on the whole training set and saved to the model path
as specified in the project_config.py

inference.py Do prediction on the test data. Predictions are saved to the output folder as csv


Files should be executed in the following order:
1. preprocessing.py
2. train_and_opt.py
3. inference.py
