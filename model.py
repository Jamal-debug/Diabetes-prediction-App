# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import pickle
df=pd.read_csv('diabetes.csv')

y=df['Outcome'].values
x=df.drop(['Outcome'],axis=1)
from sklearn.preprocessing import MinMaxScaler
scl=MinMaxScaler()
x=scl.fit_transform(x)
from sklearn.neighbors import KNeighborsClassifier
k=3
nh=KNeighborsClassifier(n_neighbors=k)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(df.drop(['Outcome'],axis=1).values,y,test_size=.15,random_state=2)
nh.fit(xtrain,ytrain)
yhat=nh.predict(xtest)

print(yhat[0:5])
pickle.dump(nh,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

