#IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

#TRAINING THE SIMPLE LINEAR REGRESSION MODEL ON THE TRAINING SET
from sklearn.linear_model import LinearRegression
regressor = LinearRegression ()
regressor.fit(x_train, y_train)

#PREDICTING THE TEST RESULTS
y_pred = regressor.predict(x_test)

#VISUALISING THE TRAINING SET RESULTS
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#VISUALISING THE TEST SET RESULTS
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue') #no need to change anything because we are visualising the test set results with the reference of tarining set results
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#FOR FINDING OUT THE SALARY OF A PERSON WHO HAS LET' SAY 15 YEARS OF EXPERIENCE
exp = input('Enter your age of experience : ')
exp = int(exp)
print('The salary of a person with 15 years of experience is : ')
print(regressor.predict([[exp]]))

#GETTING THE FINAL LINEAR REGRESSION EQUATION
print("\n")
print(regressor.coef_)
print(regressor.intercept_)

#THE EQUATION IS : SALARY = 9345.94 X YearsExperience + 26816.19