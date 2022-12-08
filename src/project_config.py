import os


#configure path
parent=os.path.dirname(os.getcwd())
input_loc=os.path.join(parent,"input")
model_path=os.path.join(parent,"models")
output_path=os.path.join(parent,"output")

call_file="Seattle_Real_Time_Fire_911_Calls.csv"
weather_file="weather_2017_2022.csv"

mem_limit=1000
starting_year=2017
last_year=2022

#configure features
train_features=['year', 'month', 'day', 'hour', 'week','PRCP', 'SNOW', 'SNWD','TMAX', 'TMIN']
target_features=["volume"]

#configure grid search and splits
time_cv_split=5
random_iter=2
perc=0.20

param_grid={
    'n_estimators': [100],
    'min_samples_split': [2],
    'min_samples_leaf': [4 ],
    'max_features': [1],
    'max_depth': [10]
}
