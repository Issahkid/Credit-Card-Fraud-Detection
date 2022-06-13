import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFS
# sns.set()
data = pd.read_csv('D:\\Desktop\\New folder\\FraudDetection\\creditcard.csv') #loading data
# print (data)
# print(data.shape) 
# print(data.describe())
# print(data.head)
# print(data.tail)
# print(data.isna)
# print(data.isnull) 
not_fraud = {0:'Not Fraud'} #find all entries that are not fraud
fraud = {1:'Fraud'} # find fraud cases
fraud_cases = len(data[data.Class==1])
non_fraud= len(data[data.Class==0])
# print(fraud_cases,non_fraud)
fraud_percentage = (fraud_cases/(fraud_cases + non_fraud))*100
# print(fraud_percentage)
scaler= StandardScaler() #normalize data
normalze = data['Normalize_amount']=scaler.fit_transform(data["Amount"].values.reshape(-1,1))
drp = data.drop(["Amount","Time"],inplace=True,axis=1) 
# print(data.head)
x = data.drop(["Class"],axis = 1) #independent variable
y = data["Class"]

trainn = x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
check_train = x_train.shape #chcek training data
check_test = x_test # check data under test
# print(check_train,check_test)
rf1 = RFS(n_estimators=100) # creating the model
training = rf1.fit(x_train,y_train) 
# print(training)
prediction_rf = rf1.predict(x_test) #predict
rf_score = rf1.score(x_test,y_test)*100 
print(rf_score)

