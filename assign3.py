# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 21:03:13 2022

@author: Kedar
"""
from sklearn import datasets
import sklearn
from sklearn import preprocessing
import joblib
from sklearn import metrics
from sklearn import model_selection
import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
import json
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st
import time
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix


# latest_iteration = st.empty()
# bar = st.progress(0)

# for i in range(100):
#   # Update the progress bar with each iteration.
#   latest_iteration.text(f'Loading Model {i+1} % Complete')
#   bar.progress(i + 1)
#   time.sleep(0.3)
# switcher = False


    
df = pd.read_csv('vineyard_weather_1948-2017.csv')
print (df.shape)
df['DATE'].dtype
df['DATE'] = pd.to_datetime(df['DATE'])
df['Year-Week'] = df['DATE'].dt.strftime('%Y-%U')
df['Week_Number'] = df['DATE'].dt.strftime('%U')

df3 = df.groupby(['Year-Week','Week_Number']).resample('W', 
                       on='DATE').agg({ 
                                        'PRCP': 'sum',
                                        'TMAX': 'sum',
                                        'TMIN': 'sum',
                                      }).reset_index().sort_values(by='DATE')  

mask = (df3['Week_Number'] > '34') & (df3['Week_Number'] <= '40')
final_data = df3.loc[mask]
final_data['Storm'] = False

mask = (final_data['PRCP']>=0.35)&(final_data['TMAX']<= 80)
final_data.loc[mask,'Storm'] = True
print(final_data.describe())
X = final_data.drop(['Storm','Year-Week','DATE'], axis=1)
y = final_data['Storm'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=0)
print(X.shape, y.shape)



scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



## GAUSIAN BAYES MODEL

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn+fp)
print(' specificity: %0.5f' % specificity )
sensitivity = tp / (tp + fn)
print(' sensitivity: %0.5f' % sensitivity )

acc =accuracy_score(y_test, y_pred)
print(' AUC: %0.5f' % acc )

pre =precision_score(y_test, y_pred)
print(' Precision: %0.3f' % pre )

recall =recall_score(y_test, y_pred)
print(' Recall: %0.3f' % recall )


def calc_payout(p_s, model_s, cost_h, cost_nh_s, cost_nh_ns):
    p_dns_ns = model_s*(1- p_s)
    p_dns = model_s*(1-p_s) + (1-model_s)*(p_s)
    p_ns_dns = p_dns_ns/p_dns
    e_val = p_dns*(cost_nh_ns*p_ns_dns + cost_nh_s*(1-p_ns_dns)) + cost_h*(1-p_dns)
    return e_val


st.text_input("Enter Chance of Botrytis :", key="botrytis")
st.text_input("Enter Chance of no Sugar increase :", key="sugar1")
st.text_input("Enter Chance of Typical Suagr increase :", key="sugar2")
st.text_input("Enter Chance of High Sugar increase :", key="sugar3")

# You can access the value at any point with:
mold = st.session_state.botrytis
sugar1 = st.session_state.sugar1
sugar2 = st.session_state.sugar2
sugar3 = st.session_state.sugar3

P_NH_NS = float(sugar1)
P_NH_TS = float(sugar2)
P_NH_HS = float(sugar3)
P_NH_M = float(mold)
P_NH_NM = 1- P_NH_M
p_s = 0.5

cost_h =12*(6000*5+2000*10+2000*15)
Cost_NH_NS = 12*(6000*5 +2000*10 + 2000*15)
Cost_NH_TS = 12*(5000*5 + 1000*10 + 2500*15 + 1500*30)
Cost_NH_HS = 12*(4000*5 + 2500*10 + 2000*15 +1000*30 + 500*40)
Cost_NH_NM = 12*(5000*5 + 1000*10)
Cost_NH_M = 12*(5000*5 + 1000*10 +2000*120)

cost_nh_s = P_NH_NM * Cost_NH_NM + P_NH_M * Cost_NH_M
cost_nh_ns = P_NH_NS *Cost_NH_NS + P_NH_TS*Cost_NH_TS + P_NH_HS *Cost_NH_HS
model_s = specificity

print("calculating E Value") 
e_val = calc_payout(p_s,model_s,cost_h,cost_nh_s,cost_nh_ns)

specificity = []
i=0
while i <= 1:
    specificity.append(i)
    i+= 0.01
estimates = [calc_payout(0.75,s,cost_h,cost_nh_s,cost_nh_ns) for s in specificity]  
clair = [cost_h - e for e in estimates]
for s in range(len(specificity)):
    if abs(clair[s]) < 0.1:
        print("Inflection point : ", specificity[s])

if e_val > cost_h:
    recc =" It is Reccomended to purchase Clairvoyance"
else:
    recc =" Harvest now without Clairvoyance"
        
if st.button('Compute E Value'):
    st.write("E Value is :"+ str(round(e_val,2)))
    st.write("Reccomendation:" + recc)



