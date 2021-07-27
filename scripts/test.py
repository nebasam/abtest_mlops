# importing libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
# Get url from Dvc
import dvc.api
path = 'data/AdSmartABdatamplat.csv'
repo = '/home/neba/Desktop/abtest_mlops'
version = 'version_two_platform'
data_url = dvc.api.get_url(
    path= path,
    repo= repo,
    rev= version
)
data = pd.read_csv(data_url,sep=',')
#log data params
mlflow.log_param('data_url',data_url)
mlflow.log_param('data_version',version)
mlflow.log_param('input_cols',data.shape[1])
mlflow.log_param('input_rows',data.shape[0])
# droping the users that did not answer
def drop_no_responds(df):
    data = df.query("not (yes == 0 & no == 0)")
    return data
cleaneddata = drop_no_responds(data)
# creating new column called aware
data['aware'] = data['yes'].map(lambda x: x==1)
# dropping yes and no column
data = data.drop(columns = ['yes', 'no', 'auction_id'], axis=1)
data.head()
# splitting data to train and test 
y = data['aware']
data = data.drop('aware', inplace=False, axis=1)
X = data
test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
y_train = pd.DataFrame(y_train)

# Log artifacts: columns used for modeling
cols_x = pd.DataFrame(list(X_train.columns))
cols_x.to_csv('features.csv', header= False, index=False)
mlflow.log_artifact('features.csv')

cols_y = pd.DataFrame(list(y_train.columns))
cols_y.to_csv('targets.csv', header= False, index=False)
mlflow.log_artifact('targets.csv')