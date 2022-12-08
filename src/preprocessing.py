
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# RAM limit for loading files, give warning if exceeded
mem_limit=1000
starting_year=2017
last_year=2022

# input directory, contains training and testing data
input_loc=os.path.join(os.path.dirname(os.getcwd()),"input")
call_data=os.path.join(input_loc,"Seattle_Real_Time_Fire_911_Calls.csv")
weather_data=os.path.join(input_loc,"weather_2017_2022.csv")
all_data=[call_data,weather_data]

#filzesize warning
for data in all_data:
    file_stats=os.stat(data)
    print('File Size in MegaBytes is {}'.format(file_stats.st_size / (1024 * 1024)))
    assert file_stats.st_size / (1024 * 1024) < mem_limit , "File too big"





def date_formatter(df,date_name,temp=["year","month","week","day","hour"]):
    '''
    create relevant time columns
    '''
    
    all_formats=["year","month","week","day","hour"]
    assert date_name in df.columns , "Date column does not exist"
    assert all([True if x in all_formats else False for x in temp]), "Non available time format"
    
    #convert to datetime object
    df[date_name]=pd.to_datetime(df[date_name])
    
    for _format in temp:
        if _format=="year":
            df[_format]=df[date_name].dt.year
        elif _format=="month":
            df[_format]=df[date_name].dt.month
        elif _format=="week":
            df[_format]=df[date_name].dt.weekday
        elif _format=="day":
            df[_format]=df[date_name].dt.day
        elif _format=="hour":
            df[_format]=df[date_name].dt.hour
    
    return df



em_data= pd.read_csv(call_data)
we_data= pd.read_csv(weather_data)

em_data=date_formatter(em_data,"Datetime",["year","month","week","day","hour"])
# only keep relevant columns
em_data.drop(["Address", "Type","Latitude", "Longitude","Report Location", 
              "Incident Number","Datetime"], axis=1,inplace=True)
em_data.dropna(inplace=True)
em_data=em_data.loc[em_data["year"]>=starting_year].copy()

# create call volume columns as label for training, coarseness set to hour
volume=em_data.value_counts(sort=False).values.copy()
em_data.drop_duplicates(inplace=True)
em_data.loc[:,"volume"]=volume




we_data=date_formatter(we_data,"DATE",["year","month","week","day"])
we_data=we_data.drop(["STATION","DAPR",'DAPR_ATTRIBUTES',"MDPR",'MDPR_ATTRIBUTES','PRCP_ATTRIBUTES',
                     'TOBS', 'TOBS_ATTRIBUTES', 'WT01', 'WT01_ATTRIBUTES', 'WT06',
       'WT06_ATTRIBUTES', 'WT11', 'WT11_ATTRIBUTES','SNOW_ATTRIBUTES','SNWD_ATTRIBUTES',
                      'TMAX_ATTRIBUTES','TMIN_ATTRIBUTES','DATE'], axis=1)
we_data.fillna(method="pad", inplace=True)

# reduce variance through log scaling
we_data["PRCP"]=we_data.loc[:,"PRCP"].apply(lambda x: np.log(1+x))
we_data["SNWD"]=we_data.loc[:,"SNWD"].apply(lambda x: np.log(1+x))



merged_data=em_data.copy()
merged_data=merged_data.merge(we_data, how="outer")
merged_data.sort_values(by=['year','month','day','hour'], inplace=True)
merged_data.fillna(method="pad", inplace=True)
merged_data.reset_index(drop=True, inplace=True)



train_data=merged_data.loc[merged_data["year"]<last_year]
test_data=merged_data.loc[merged_data["year"]==last_year]


train_data.to_csv(os.path.join(input_loc,"train.csv"), index=False)
test_data.to_csv(os.path.join(input_loc,"test.csv"), index=False)
