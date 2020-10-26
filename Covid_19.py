# -*- coding: utf-8 -*-
#Author: Kevin Garcia
#Project Covid-19 
#Date: 06/14/2020


#https://www1.nyc.gov/site/doh/covid/covid-19-data.page
#https://github.com/nychealth/coronavirus-data
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import io
import requests
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import statsmodels.formula.api as smf

#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()



Token = {'api_key':'eaf892facfe009647bd06ce2db699090fa084edb'}

#Github username
username = "kgarcia07"

temp = "https://raw.githubusercontent.com/nychealth/coronavirus-data/master/"
#Create a dictionary database
dictionary = {"Age":"by-age.csv", "Borough":"by-boro.csv", "Poverty":"by-poverty.csv",
              "Race":"by-race.csv", "Sex":"by-sex.csv", "Case":"case-hosp-death.csv",
              "Testing":"tests.csv","Hosp_Visits":"syndromic_data.csv", 
              "Global":"data-by-modzcta.csv"}

list = []
for i in dictionary:
    list.append(i)

print(list)


data = pd.DataFrame()
get_data = 0
while get_data == 0:
    user_input = input("Which data would you like to use? or type no when finished: ") 
    
    if user_input == "no":
        break
    elif user_input in list:
        #Url to request
        url = temp + dictionary[user_input]
        #Make the request and return the json
        user_data = requests.get(url).content

        #Pretty print JSON data
        address = pd.read_csv(io.StringIO(user_data.decode('utf-8')))
        address.head()
        count = 0
        while count == 0:
            feature_input = eval(input("\nWhich columns would you like to append? or 100 when finished: "))
            if feature_input == 100:
                count = 1
            else:       
                #var is a temporary list
                address = address.loc[(address.iloc[:,0] >= '03/03/2020') & (address.iloc[:,0] <= '09/23/2020')]
                address.reset_index(drop = True, inplace = True)
                data = pd.concat([data, address.iloc[:, feature_input]], axis = 1)
                print(data)


def normalize(dataFrame):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

df = data
df = normalize(df)
#Create a ordinary least squares model 
model = smf.ols(formula= 'DEATH_COUNT ~ POSITIVE_TESTS + CASE_COUNT', data=df)
results_formula = model.fit()
results_formula.params
print(results_formula.summary())


## Prepare the data for Visualization
x_surf, y_surf = np.meshgrid(np.linspace(df.POSITIVE_TESTS.min(), df.POSITIVE_TESTS.max(), 100),
                             np.linspace(df.CASE_COUNT.min(), df.CASE_COUNT.max(), 100))
onlyX = pd.DataFrame({'POSITIVE_TESTS': x_surf.ravel(), 'CASE_COUNT': y_surf.ravel()})
fittedY = results_formula.predict(exog = onlyX)

## convert the predicted result in an array
fittedY = np.array(fittedY)

# Visualize the Data for Multiple Linear Regression

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df['POSITIVE_TESTS'],df['CASE_COUNT'],df['DEATH_COUNT'],c = 'red',marker = 'o',alpha=0.6)
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape),color = 'b',alpha=0.3)
ax.set_xlabel('POSITIVE_TESTS')
ax.set_ylabel('CASE COUNT')
ax.set_zlabel('DEATH COUNT')
ax.set_title('Covid-19 Regression Model')
plt.show()
"""
df = data
df.to_excel(excel_writer = "Covid19_Data.xlsx")
"""

