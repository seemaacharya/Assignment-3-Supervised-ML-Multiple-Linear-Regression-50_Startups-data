# -*- coding: utf-8 -*-
"""
Created on Sun May  9 20:30:30 2021

@author: DELL
"""
#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading the dataset
Startup = pd.read_csv("50_Startups.csv")
Startup.columns

#Visualization
plt.boxplot(Startup["Profit"])
plt.boxplot(Startup["R&D Spend"])
plt.boxplot(Startup["Administration"])
plt.boxplot(Startup["Marketing Spend"])

#Creating dummies for State column
Startup1 = pd.get_dummies(Startup["State"])
Startup_data = pd.concat([Startup,Startup1],axis=1)
Startup_data = Startup_data.drop(["State"],axis=1)
Startup_data = Startup_data.iloc[:,[3,0,1,2,4,5,6]]

#Correlation
Correlation = Startup_data.corr()

#Scatter plot between the variables along with the histogram
import seaborn as sns
sns.pairplot(Startup_data)

#Renaming the variable (as the big names)
Startup_data.rename(columns={'R&D Spend':'RnD','Marketing Spend':'Marketing','New York':'NewYork'},inplace= True)
#Model building
import statsmodels.formula.api as smf
model1 = smf.ols("Profit~RnD+Administration+Marketing+California+Florida+NewYork",data=Startup_data).fit()
model1.params
model1.summary()
#Here R-Squared=0.95, Adj. R Squared=0.945
#Administration and Marketing both are insignificant

#Building model individually 
model1 = smf.ols("Profit~Administration",data=Startup_data).fit()
model1.summary()
#Administration is insignificant (p=0.162)

model1 = smf.ols("Profit~Marketing",data=Startup_data).fit()
model1.summary()
#Mraketing is significant(P=0.00)

model1 = smf.ols("Profit~Administration+Marketing",data=Startup_data).fit()
model1.summary()
#Administration is significant(p=0.017) and Marketing is significant(p=0.00)

#Plotting the influence plot
import statsmodels.api as sm
sm.graphics.influence_plot(model1)

#Removing the 19,
Startup_data1= Startup_data.drop(Startup_data.index[[19]],axis=0)
model2 = smf.ols("Profit~RnD+Administration+Marketing+California+Florida+NewYork",data=Startup_data1).fit()
model2.summary()
#both Administration and Marketing are insignificant
#Removing the 19 and 49,
Startup_data2= Startup_data.drop(Startup_data.index[[19,49]],axis=0)
model3= smf.ols("Profit~RnD+Administration+Marketing+California+Florida+NewYork",data=Startup_data2).fit()
model3.summary()
#Administration is insignificant

# calculating the VIF for the independent variables
rsq_rnd = smf.ols("RnD~Administration+Marketing+California+Florida+NewYork",data= Startup_data2).fit().rsquared
vif_rnd = 1/(1-rsq_rnd)
#2.602
rsq_administration = smf.ols("Administration~RnD+Marketing+California+Florida+NewYork",data= Startup_data2).fit().rsquared
vif_administration= 1/(1-rsq_administration)
#1.148
rsq_marketing = smf.ols("Marketing~RnD+Administration+California+Florida+NewYork",data= Startup_data2).fit().rsquared
vif_marketing = 1/(1-rsq_marketing)
#2.51

#Here, all the vif's are below 10, no dependency among the input variables.

#Added Variable plot
sm.graphics.plot_partregress_grid(model2)
#As the correlation b/w the Profit and Administration is low, so let's remove the Administration variable
model4= smf.ols("Profit~RnD+Marketing+California+Florida+NewYork",data= Startup_data2).fit()
model4.summary()
#All the variables are significant
Final_model = smf.ols("Profit~RnD+Marketing+California+Florida+NewYork",data=Startup_data2).fit()
Final_model.summary()
#Here R Squared=0.96, Adj R aquared= 0.95 and all the variables are significant
pred = Final_model.predict(Startup_data2)
pred

#Added variable plot for the final model
sm.graphics.plot_partregress_grid(Final_model)

#Evaluation of the Final model
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(Startup_data2.Profit,pred))
rmse
#7398
#Visualization by scatter plot b/w Profit and predicted profit
plt.scatter(Startup_data2.Profit,pred,color = "red");plt.xlabel("Profit");plt.ylabel("Predicted Profit")
#Scatter plot b/w predicted values and residuals
plt.scatter(pred,Final_model.resid_pearson,color="green");plt.axhline(y=0,color='blue');plt.xlabel("pred");plt.ylabel("residuals")
