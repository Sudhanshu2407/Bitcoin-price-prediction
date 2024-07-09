#Importing the important libraries.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import warnings 

warnings.filterwarnings("ignore")

#------------------------------------------
#Now we read the dataset.

bitcoin_df=pd.read_csv(r"C:\sudhanshu_projects\project-task-training-course\bitcoin_price_forecasting.csv")

#------------------------------------------
#Here we split the date column into year,month and day column.

split=bitcoin_df["Date"].str.split("-",expand=True)

#------------------------------------------
#Here we create the three new column.

bitcoin_df["Year"]=split[0].astype("int")
bitcoin_df["month"]=split[1].astype("int")
bitcoin_df["day"]=split[2].astype("int")

#------------------------------------------
#Now Here we drop the date column.

bitcoin_df.drop("Date",axis=1,inplace=True)

#------------------------------------------
#Now Her we create target column.

bitcoin_df["target"]=bitcoin_df["Close"].shift(-1)

#------------------------------------------
#Here we drop the last row,because that contain nan value.

bitcoin_df.dropna(inplace=True)

#------------------------------------------
#Here we create is_quarter_end column.

bitcoin_df['is_quarter_end'] = np.where(bitcoin_df['month']%3==0,1,0)

#------------------------------------------
#Here we create two more columns.

#---------------------------------
#Here we create open-close column.
bitcoin_df["open-close"]=bitcoin_df["Open"]-bitcoin_df["Close"]

#---------------------------------
#Here we create low-high column.
bitcoin_df["low-high"]=bitcoin_df["Low"]-bitcoin_df["High"]

#-------------------------------------------
#Here we not choose the dependent and independent feature/variable.

x=bitcoin_df[["open-close","low-high","is_quarter_end"]] #Independent feature.

y=bitcoin_df["target"] #Dependent feature/variable.

#--------------------------------------------
#Now here we split the dataset into train and test dataset.

#---------------------------------
#Here we import the library.
from sklearn.model_selection import train_test_split

#--------------------------------
#Here we build x train/test and y train/test.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#--------------------------------------------
#Now here we check the description of x.

x.describe()

#Conclusion: Here we get min,max and quarter of x.

#---------------------------------------------
#Now here we do the scaling of data using standard scaler.

#--------------------------------------------
#Now, here we import the library.
from sklearn.preprocessing import StandardScaler

#-----------------------------------
#Here we create standardscaler model object.
sc=StandardScaler()

#-----------------------------------
#Here we fit and transform x_train.
x_train_sc=sc.fit_transform(x_train)

#-----------------------------------
#Here we transform x_test.
x_test_sc=sc.transform(x_test)

#------------------------------------------------
#Now here we do the normalization of data using Normalizer.

#------------------------------------
#Here we import Normalizer library.
from sklearn.preprocessing import Normalizer

#------------------------------------
#Now here we create normalizer model object.
nz=Normalizer()

#------------------------------------
#Now here we fit and transform x_train.
x_train_nz=nz.fit_transform(x_train)

#------------------------------------
#Now here we transform x_test.
x_test_nz=nz.transform(x_test)

#--------------------------------------------------
#Now here we use regression models.

#-----------------------------------------
#Here we use linear regression model.

#-----------------------------------------
#Here we import linear regression model.
from sklearn.linear_model import LinearRegression

#-----------------------------------------
#Here we create linear regression model object.
lr=LinearRegression()

#-----------------------------------------
#Here we train the lr model.
lr.fit(x_train,y_train)

#-----------------------------------------
#Here we predict the values using lr model.
y_pred_lr=lr.predict(x_test)

#-----------------------------------------
#Here we calculate accuracy of lr model.
score_lr=lr.score(x_test,y_test)
print(f"The accuracy of lr model is {score_lr}.")

#Conclusion: Here the accuracy of lr model is very less.(i.e 0.3%).

#------------------------------------------------------
#Here we try to implement the lr model on scaled data.

#------------------------------------
#Here we build linear regression model object.
lr1=LinearRegression()

#------------------------------------
#Here we train the model.
lr1.fit(x_train_sc,y_train)

#------------------------------------
#Here we predict the values using lr1 model.
y_pred_lr1=lr1.predict(x_test_sc)

#------------------------------------
#Here we calculate the accuracy of lr1 model.
score_lr1=lr1.score(x_test_sc,y_test)
print(f"The accuracy of lr1 model is {score_lr1}.")

#Conclusion: Here we seen the accuracy not improves so much.

#---------------------------------------------------------
#Now here we try to implement lr model on normalized data.

#--------------------------------
#Here we build linear regression model object.
lr2=LinearRegression()

#--------------------------------
#Here we train the lr2 model.
lr2.fit(x_train_nz,y_train)

#-------------------------------
#Here we predict the values using lr2 model. 
y_pred_lr2=lr2.predict(x_test_nz)

#-------------------------------
#Now here we calculate accuracy of lr2 model.
score_lr2=lr2.score(x_test_sc,y_test)
print(f"The accuracy of lr2 model is {score_lr2}.")

#Conclusion: Here the accuracy decreases.

#------------------------------------------------------------
#Now here we try to use regularization technique.

#----------------------------------------
#Here we import the regularization technique.
from sklearn.linear_model import Lasso,Ridge

#----------------------------------------
#Here we use lasso model.

#-----------------------------------
#Here we build lasso model object.
ls=Lasso()

#-----------------------------------
#Here we train the model.
ls.fit(x_train,y_train)

#-----------------------------------
#Here we predict the values using ls model.
y_pred_ls=ls.predict(x_test)

#-----------------------------------
#Here we find the accuracy of ls model.
score_ls=ls.score(x_test,y_test)
print(f"The accuracy of ls model is {score_ls}.")

#Conclusion: Here also the accuracy is very less.

#--------------------------------------------
#Here we use ridge model.

#-----------------------------------
#Here we create ridge model object.
rd=Ridge()

#-----------------------------------
#Here we train the model.
rd.fit(x_train,y_train)

#-----------------------------------
#Here we predict the values using rd model.
y_pred_rd=rd.predict(x_test)

#-----------------------------------
#Here we find the accuracy of rd model.
score_rd=rd.score(x_test,y_test)
print(f"The accuracy of rd model is {score_rd}.")

#Conclusion: Here also the accuracy is very less.

#---------------------------------------------------
#Now here we use non-linear regression model.

#--------------------------------------
#Here we import the polynomial regression model.
from sklearn.preprocessing import PolynomialFeatures

#--------------------------------------
#Here we create polynomial regression model object. 
poly_reg=PolynomialFeatures(degree=2)

#--------------------------------------
#Here we fit and transform x_train.
x_train_poly=poly_reg.fit_transform(x_train)

#--------------------------------------
#Here we transform the x_test.
x_test_poly=poly_reg.transform(x_test)

#---------------------------------------
#Here we create linearregression model object.
lr3_poly=LinearRegression()

#---------------------------------------
#Here we train the model.
lr3_poly.fit(x_train_poly,y_train)

#---------------------------------------
#Here we predict the values using lr3_poly model.
y_predict_lr3_poly=lr3_poly.predict(x_test_poly)

#-------------------------------------------------
#Here we import the important libraries.
from sklearn.metrics import r2_score,mean_squared_error

#-------------------------------------------------
#Here we find the accuracy of model.
score_lr3_poly=r2_score(y_test,y_predict_lr3_poly)
print(f"The accuracy of score_lr3_poly is {score_lr3_poly}.")

#Here also the accuracy is very less.

#----------------------------------------------------
#Now here we use svr(support verctor regressor) model.

#--------------------------------------
#Here we import svr model.
from sklearn.svm import SVR

#--------------------------------------
#Here we build svr model object.
svr=SVR(gamma="auto",degree=4)

#--------------------------------------
#Here we train the svr model.
svr.fit(x_train,y_train)

#--------------------------------------
#Here we predict the values using svr model.
y_pred_svr=svr.predict(x_test)

#--------------------------------------
#Here we find the accuracy of svr model.
score_svr=svr.score(x_test,y_test)
print(f"The accuracy of svr model is {score_svr}.")

#Conclusion: Here the accuracy is also very less.

#----------------------------------------------------------
#Now here we use knn model.

#-------------------------------------------------
#Here we import the knn model.
from sklearn.neighbors import KNeighborsRegressor

#-------------------------------------------------
#Here we build the knn model object.
knn=KNeighborsRegressor(n_neighbors=1)

#-------------------------------------------------
#Here we train the model.
knn.fit(x_train,y_train)

#-------------------------------------------------
#Here we predict the values using knn model.
y_pred_knn=knn.predict(x_test)

#-------------------------------------------------
#Here we find the accuracy of knn model.
score_knn=knn.score(x_test,y_test)
print(f"The accuracy of knn model is {score_knn}.")

#Conclusion: Here also the accuracy is very less.

#-----------------------------------------------------------
#Here we use decision tree regressor model.

#-------------------------------------------------
#Here we import decision tree regressor model.
from sklearn.tree import DecisionTreeRegressor

#-------------------------------------------------
#Here we create decision tree regressor model object.
dtr=DecisionTreeRegressor()

#-------------------------------------------------
#Here we train the model.
dtr.fit(x_train,y_train)

#-------------------------------------------------
#Here we predict the values using dtr model.
y_pred_dtr=dtr.predict(x_test)

#_------------------------------------------------
#Here we find the accuracy of dtr model.
score_dtr=dtr.score(x_test,y_test)
print(f"The accuracy of dtr model is {score_dtr}.")

#Conclusion: Here also the accuracy of dtr model is very less.

#------------------------------------------------------------
#Here we use random forest regressor model.

#-------------------------------------------------
#Here we import regressor forest regressor model.
from sklearn.ensemble import RandomForestRegressor

#-------------------------------------------------
#Here we create random forest regressor model object.
rfr=RandomForestRegressor()

#-------------------------------------------------
#Here we train the model.
rfr.fit(x_train,y_train)

#-------------------------------------------------
#Here we predict the values using rfr model.
y_pred_rfr=rfr.predict(x_test)

#-------------------------------------------------
#Here we find the accuracy of rfr model.
score_rfr=rfr.score(x_test,y_test)
print(f"The accuracy of rfr model is {score_rfr}.")

#Conclusion: Here also the accuracy is very less.

#------------------------------------------------------------
#Now we implement rfr model on scaled data.

#--------------------------------------------
#Here we build rfr1 model object.
rfr1=RandomForestRegressor()

#--------------------------------------------
#Here we train the model.
rfr1.fit(x_train_sc,y_train)

#--------------------------------------------
#Here we predict the values using rfr1 model.
y_pred_rfr1=rfr1.predict(x_test_sc)

#--------------------------------------------
#Here we find the accuracy of rfr1 model.
score_rfr1=rfr1.score(x_test_sc,y_test)
print(f"The accuracy of rfr1 model is {score_rfr1}.")

#Conclusion: But accuracy not increases.