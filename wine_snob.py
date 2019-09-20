import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

print(data.head())

#Trainging and data spliting.
y = data.quality 
X = data.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y) #remember to stratify the data based on 'y'
#intro to the transformer used to save the scales of mean and standard deviations
scaler = preprocessing.StandardScaler().fit(X_train)
#Applying the transformer unto the test data
X_test_scaled = scaler.transform(X_train)
print(X_test_scaled.mean(axis=0))
print(X_test_scaled.mean(axis=0))

#pipelining with preprocessing and model
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))
