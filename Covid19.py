#!/usr/bin/env python
# coding: utf-8

# # Covid19
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import datetime as dt
from datetime import date
import matplotlib.dates as mdates
import matplotlib.patches as mpatches


# In[2]:


egy_covid=pd.read_csv("EGY-covid19.csv")


# In[3]:



X= egy_covid.iloc[:,3].values
y=egy_covid.iloc[:,5].values

y1=egy_covid.iloc[0:200,5].values
y2=egy_covid.iloc[200:400,5].values
y3=egy_covid.iloc[400:600,5].values


# In[4]:




numbers =[1]
lablels=['14/2/2020', '14/3/2020', '14/4/2020', '14/5/2020','14/6/2020','14/7/2020'
         ,'14/8/2020','14/9/2020','14/10/2020','14/11/2020','14/12/2020','14/1/2021','14/2/2021','14/3/2021'
        ,'14/4/2021','14/5/2021','14/6/2021','14/7/2021','14/8/2021','14/9/2021']
i=3
while(i<6):
    
    d0 = date(2020, 2, 14)
    d1 = date(2020, i, 14)
    delta = d1 - d0
    w=str(delta)
    w=int(w[0]+w[1])
    w=w+1
    #print(d1," - ",d0," = ",w)
    numbers.insert(len(numbers),w)
    i=i+1
    
while(i<13):
    d0 = date(2020, 2, 14)
    d1 = date(2020, i, 14)
    delta = d1 - d0
    w=str(delta)
    w=int(w[0]+w[1]+w[2])
    w=w+1
    #print(d1," - ",d0," = ",w)
    numbers.insert(len(numbers),w)
    i=i+1

i=1
while(i<10):
    d0 = date(2020, 2, 14)
    d1 = date(2021, i, 14)
    delta = d1 - d0
    w=str(delta)
    w=int(w[0]+w[1]+w[2])
    w=w+1
    #print(d1," - ",d0," = ",w)
    numbers.insert(len(numbers),w)
    i=i+1
    


# In[5]:


plot = egy_covid.plot(x='date', y='new_cases', style='o',figsize=(20, 5))
plt.title('new cases every day')
plt.xlabel('date ')
plt.ylabel('new cases')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())

plt.gcf().autofmt_xdate()
plt.xticks(numbers,lablels)
plt.show()


# In[6]:


X=X.reshape(-1,1)
y=y.reshape(-1,1)

y1=y1.reshape(-1,1)
y2=y2.reshape(-1,1)
y3=y3.reshape(-1,1)
print(X.shape)


# In[7]:


x=np.empty([616, 1],dtype=float)
x1=np.empty([200, 1],dtype=float)
x2=np.empty([200, 1],dtype=float)
x3=np.empty([200, 1],dtype=float)
for i in range(200):
    x1[i][0]=i+1
    x2[i][0]=i+1
    x3[i][0]=i+1
for i in range(616):
    x[i][0]=(i+1)           #int(my_list[0][:])
    
#print(x[0][3])


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state = 0 , shuffle= False)


# In[9]:


# Fitting Polynomial Regression to the dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score
for i in range(100):
    poly_reg = PolynomialFeatures(degree =i)
    X_poly = poly_reg.fit_transform(x1)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y1)
    
    
    X_poly = poly_reg.fit_transform(x2)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y2)
    
    X_poly = poly_reg.fit_transform(x3)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y3)


# In[10]:


# calc the error


poly_reg = PolynomialFeatures(degree =8)
X_poly = poly_reg.fit_transform(x1)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y1)
    
    
X_poly = poly_reg.fit_transform(x2)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y2)
    
X_poly = poly_reg.fit_transform(x3)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y3)
    


# In[11]:


lablels=['14/2/2020', '14/3/2020', '14/4/2020', '14/5/2020','14/6/2020','14/7/2020'
         ,'14/8/2020','14/9/2020','14/10/2020','14/11/2020','14/12/2020','14/1/2021','14/2/2021','14/3/2021'
        ,'14/4/2021','14/5/2021','14/6/2021','14/7/2021','14/8/2021','14/9/2021','14/10/2021','14/11/2021','14/12/2021',
        '14/1/2022','14/2/2022','14/3/2022']
i=10
while(i<13):
    d0 = date(2020, 2, 14)
    d1 = date(2021, i, 14)
    delta = d1 - d0
    w=str(delta)
    w=int(w[0]+w[1]+w[2])
    w=w+1
    #print(d1," - ",d0," = ",w)
    numbers.insert(len(numbers),w)
    i=i+1


i=1
while(i<4):
    d0 = date(2020, 2, 14)
    d1 = date(2022, i, 14)
    delta = d1 - d0
    w=str(delta)
    w=int(w[0]+w[1]+w[2])
    w=w+1
    #print(d1," - ",d0," = ",w)
    numbers.insert(len(numbers),w)
    i=i+1
    


# In[12]:


x_PRETECT=np.empty([200, 1],dtype=float)
for i in range(200):
    x_PRETECT[i][0]=i+1   
    
#x1=x1

plt.scatter(x, y, color = 'red')
plt.plot(x_PRETECT+600, lin_reg_2.predict(poly_reg.fit_transform(x_PRETECT)), color = 'blue')
plt.title ('egy-covid19')
plt.xlabel('day')
plt.ylabel('new cases')
plt.show()
#print(lin_reg_2.predict(poly_reg.fit_transform(x_PRETECT)))


# In[13]:


y_pretect=lin_reg_2.predict(poly_reg.fit_transform(x_PRETECT))
end_date=0
peak_date=0
for i in range(200):
    if y_pretect[end_date][0] > y_pretect[i][0]:
        end_date=i
    if y_pretect[peak_date][0] < y_pretect[i][0]:
        peak_date=i
   # print(y_pretect[i][0])
    
end_date=end_date+600
peak_date=peak_date+600


# In[14]:


print(peak_date)
print(end_date)
#print(y_pretect[end_date][0])


# In[15]:



d0 = date(2020, 2, 14)
d1= d0 + dt.timedelta(days=peak_date)
d2= d0 + dt.timedelta(days=end_date)
print(str(d0+dt.timedelta(days=600)))
print("the peak_date in the 4th wave is : "+str(d1))
print("the last day in the 4th wave is : "+str(d2))


# In[16]:



fig= plt.figure(figsize=(20,5))

axes= fig.add_axes([0.1,0.1,0.8,0.8])


plt.plot(x_PRETECT+600, lin_reg_2.predict(poly_reg.fit_transform(x_PRETECT)), color = 'blue')

plt.scatter(x, y, color = 'red')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())

plt.gcf().autofmt_xdate()
plt.xticks(numbers,lablels)
plt.title ('egy-covid19')
plt.xlabel('date')
plt.ylabel('new cases')

red_patch = mpatches.Patch(color='red', label='old cases')
blue_patch = mpatches.Patch(color='blue', label='predicted cases')

plt.legend(handles=[red_patch,blue_patch])

plt.show()


# In[ ]:




