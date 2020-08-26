import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import streamlit as st


st.title('Corona Cases Predictor')

## Load Data ##
data = pd.read_csv('coronaCases.csv',sep=',')
data = data[['id','cases']]
print('-'*30); print('Heading'); print('-'*30)
print(data.head())

## Prepare Data ##
print('-'*30); print('Prepare Data'); print('-'*30)
x = np.array(data['id']).reshape(-1,1)
y = np.array(data['cases']).reshape(-1,1)

st.header('Plotting Dataset')
plt.plot(y,'-m')
st.pyplot()

polyFeat = PolynomialFeatures(degree=3)
x = polyFeat.fit_transform(x)

## Training Data ##
print('-'*30); print('Training Data'); print('-'*30)
model = linear_model.LinearRegression()
model.fit(x,y)

## Saving Model ##
filename = 'trained_lr_model.sav'
# pickle.dump(model,open(filename,'wb'))

loaded_model = pickle.load(open(filename,'rb'))
accuracy = model.score(x,y)
print(f'Accuracy: {round(accuracy*100,3)}%')
st.header(f'Accuracy: {round(accuracy*100,3)}%')

y0 = model.predict(x)

## Prediction ##
print('-'*30); print('Prediction'); print('-'*30)
days = int(st.sidebar.text_input('Prediction for number of days '))
st.sidebar.text('* Red in Final Plot means Prediction')
#print(f'Prediction - Cases after {days} days: ', end='')
st.header(f'Prediction - Cases after {days} days')
prediction = round(int(model.predict(polyFeat.fit_transform([[234+days]])))/1000000,2)
#print(prediction,'Million')
st.title(f'{prediction} Million')

st.header('Output')
x1 = np.array(list(range(1,234+days))).reshape(-1,1)
y1 = model.predict(polyFeat.fit_transform(x1))
plt.plot(y1,'--r',label='Prediction')
plt.plot(y0,'--b',label='Previous Data')
plt.xlabel('Days')
plt.ylabel('Cases')
plt.title('Final Plot')
st.pyplot()
# plt.show()

