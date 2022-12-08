
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from model import RF_Model

parent=os.path.dirname(os.getcwd())
input_loc=os.path.join(parent,"input")
model_path=os.path.join(parent,"models")

train_features=['year', 'month', 'day', 'hour', 'week','PRCP', 'SNOW', 'SNWD','TMAX', 'TMIN']
target_features=["volume"]



model=RF_Model()
model.load_model()
_,_, test_x, test_y=model.load_train_test(train_features=train_features,
                                                     target_features=target_features,path=input_loc)

score=model.evaluate(test_x, test_y)
print("MAE on test set ", score)



m_pred=model.predict_(test_x[test_x['month']==1])
m_target=test_y[test_x['month']==1]
week=test_x.loc[test_x['month']==1,"week"]

w_pred=model.predict_(test_x[(test_x['month']==1) & (test_x["day"].between(0,6)) ])
w_target=test_y[(test_x['month']==1) & (test_x["day"].between(0,6)) ]
hours=test_x.loc[(test_x['month']==1) & (test_x["day"].between(0,6)), "hour"]


fig, ax=plt.subplots(2)
ax=ax.ravel()

ax[0].plot(m_pred, label='prediction')
ax[0].plot(m_target, alpha=0.5, label='target')
ax[0].set_xticks(range(0,len(m_pred),24*7))
ax[0].set_xticklabels(week.values[::24*7])
ax[0].set_xlabel("week")
ax[0].legend(bbox_to_anchor=(1., 1.0))
ax[0].set_title("MAE={} ,std={}".format(np.around(np.abs(m_pred-m_target).mean(),2),
                                        np.around(np.std(np.abs(m_pred-m_target)),2 )))

ax[1].plot(w_pred, label='prediction')
ax[1].plot(w_target,alpha=0.5, label='target')
ax[1].set_xticks(range(0,len(w_pred),6))
ax[1].set_xticklabels(hours.values[::6])
ax[1].set_xlabel("hour")
ax[1].legend(bbox_to_anchor=(1., 1.0))
ax[1].set_title("MAE={} ,std={}".format(np.around(np.abs(w_pred-w_target).mean(),2),
                                        np.around(np.std(np.abs(w_pred-w_target)),2 )))

ax[0].set_ylabel("call volume")
ax[1].set_ylabel("call volume")


plt.tight_layout()
plt.show()
