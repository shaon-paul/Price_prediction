# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 23:40:03 2020

@author: paul
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("PotatoPrice.csv") 

df

plt.xlabel('Potato in kilogram(kg)')
plt.ylabel('price in Rupees')
plt.scatter(df.potato_kg, df.price)

X = df[['potato_kg']]
y = df['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10) 
#if you use "random_state=10" then the smaple will be same all the time

X_train
X_test
y_train
y_test

# use the regression model for the dataset
reg=LinearRegression() #creat the object for the regression
reg.fit(X_train, y_train)  #pass the data through the model, reg.fit(1st argument, 2nd argument);
            #1st argument have to be two dimentional or 2D array
            #2nd argument have to be y axis or the output, since y=mx+c
            
reg.predict(X_test)
y_test

#We will find the accuracy of this model(our model was liner regression model) for our dataset
reg.score(X_test, y_test)


# Give any unknown potato kilogram value,to know the price
#(N.B: the potato kilogram value have to be any value upto 1,for the decent prediction. Since our fitted data potato_kg range is 1 to 7)
reg.predict([[1.1505659]])

#Simple user interface to run our model the model
x=input('To know the potato price,Enter the potato killogram upto 1 : ')


array = np.array(x) #input converted into 1 dimentional array
fvalu = array.astype(np.float) # 1 dimentional array into 1 dimentional float array
fvalu_2D=([[fvalu]]) # 1 dimentional array to 2 dimentional array
#print(fvalu_2D)

my_prediction=reg.predict(fvalu_2D)
#print(my_prediction)

#price=np.asscalar(np.array(my_prediction)) #convert vector into scalar using this one line only

#convert vector into scalar using below two lines
price=np.array(my_prediction) 
price=price.item()

print('So',x,' killogram potato price is =',price ,' Rupees')