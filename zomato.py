# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:41:47 2020

@author: sahil
"""
#importing libraries
import pandas as pd
import numpy as np

#importing dataset
zom= pd.read_excel("H:\\all datasets\\zomato_train.xlsx")

#dataset details
zom.shape
zom.info()
zom.describe()

#E.D.A

#total number of resturants
zom['name'].nunique() #there are 8487 resturants
#most number of resturants is of
zom.name.value_counts() [:25].plot(kind='bar') #cafe coffee day is the max
#resturants accepting online orders
zom.online_order.value_counts().plot(kind='bar') #yes=24330 no=17043
#resturants accepting pre-booking of tables
zom.book_table.value_counts().plot(kind='bar') #no= 36231 yes=5142
#resturants located in areas
zom['location'].nunique() #there are 93 loations 
#higest number of resturants
zom.location.value_counts() [:25].plot(kind='bar') #BTM has the highest no. of resturants
#types of resturants
zom['rest_type'].nunique() #there are 93 types of restaurants
#top resturant type
zom.rest_type.value_counts() [:25].plot(kind='bar') #Quick bites is the  most frequently visited
#kinds of dish
zom['dish_liked'].nunique() #there are 5026 kinds of dish
#most liked dish
zom.dish_liked.value_counts() [:25].plot(kind='bar') #biryani is the most liked dish
#types of cuisines
zom['cuisines'].nunique() #there are 2654 types of cuisines
#most liked cuisines 
zom.cuisines.value_counts() [:25].plot(kind='bar') #North indian is the most preffered cuisines
#renaming the listed_in(type) 
zom.rename(columns={'listed_in(type)':'listedin_type'}, inplace=True)
#most of the resturants are giving the option of delivery
zom.listedin_type.value_counts().plot(kind='bar') #20778 resturants are delivering food
#renaming the approximate cost for two columns 
zom.rename(columns={'approx_cost(for two people)': 'avg_cost'}, inplace=True)
#approximate cost for 2 peoples
zom.avg_cost.value_counts()[:25].plot(kind='bar') #300 is the average cost for 2 peoples
#average rating
zom.rates.value_counts()[:25].plot(kind='bar') #3.9 is the average rating

#missing values in the data
zom.isnull().sum() #dish_liked has the maximum missing values

#dropping the missing values from the cuisine column
zomato= zom[zom.cuisines.isna()==False]
#checking for missing values again
zomato.isnull().sum()

#dropping the unnessary colums 
zomato.drop(columns=['url','address','phone','listed_in(city)','dish_liked','reviews_list','menu_item','listedin_type'], inplace  =True)

#replacing missing values
zomato['rates'] = zomato['rates'].replace('-',np.NaN)
zomato["rates"].fillna(3.9,inplace=True)
zomato['rest_type'] = zomato['rest_type'].str.replace(',' , '') 
zomato["rest_type"].fillna('Quick Bites',inplace=True)
zomato["avg_cost"] = zomato["avg_cost"].str.replace(',' , '') 
zomato["avg_cost"].fillna(300,inplace=True)

zomato.info()

#pre process for model building
zomato["rates"] = zomato["rates"].astype('float')
zomato["avg_cost"] = zomato["avg_cost"].astype('float')
zomato['online_order']= pd.get_dummies(zomato.online_order, drop_first=True)
zomato['book_table']= pd.get_dummies(zomato.book_table, drop_first=True)
zomato['rest_type'] = zomato['rest_type'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
zomato['cuisines'] = zomato['cuisines'].str.replace(',' , '') 
zomato['cuisines'] = zomato['cuisines'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))

#Label encoding
from sklearn.preprocessing import LabelEncoder
T = LabelEncoder()                 
zomato['location'] = T.fit_transform(zomato['location'])
zomato['rest_type'] = T.fit_transform(zomato['rest_type'])
zomato['cuisines'] = T.fit_transform(zomato['cuisines'])

x = zomato.drop(['rates','name'],axis = 1)
y = zomato['rates']

#splitting the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 123)

#model building

#linear regression model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)
lr.score(X_test,y_test)*100 #20.18% accuracy

#Ridge model
from sklearn.linear_model import Ridge
rdg = Ridge()
rdg.fit(X_train,y_train)
y_pred_rdg = rdg.predict(X_test)
rdg.score(X_test,y_test)*100 #20.18% accuracy

#Lasso model
from sklearn.linear_model import Lasso
ls = Lasso()
ls.fit(X_train,y_train)
y_pred_ls = ls.predict(X_test)
ls.score(X_test,y_test)*100 #17.53% accuracy

#Random Forest 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
rf.score(X_test,y_test)*100 #accuracy = 89.26%

#predection for random forest regression
Randpred = pd.DataFrame({ "actual": y_test, "predicted": y_pred_rf })
Randpred 