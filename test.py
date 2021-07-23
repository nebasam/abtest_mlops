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
path = 'data/AdSmartABdata1ml.csv'
repo = '/home/neba/Desktop/abtest_mlops'
version = 'V_browser'
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
