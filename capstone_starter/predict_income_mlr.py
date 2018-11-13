# Predicts income based on education, drugs and sex, using LinearRegression model.

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from time import time

def generate_column_mapping(column_values):
    unique_column_values = list(set(column_values))
    column_mapping = {}
    index = 0
    for val in unique_column_values:
        column_mapping[val] = index
        index += 1
    return column_mapping

def get_data_set(data_frame):
    #Removing the NaNs
    X_columns = ["education","drugs","sex"]
    data_frame[X_columns] = data_frame[X_columns].replace(np.nan, '', regex=True)
    data_frame["income"] = data_frame["income"].replace(np.nan, -1, regex=True)
    #Augment Data
    data_frame["education_code"] = data_frame.education.map(generate_column_mapping(data_frame["education"]))
    data_frame["drugs_code"] = data_frame.drugs.map(generate_column_mapping(data_frame['drugs']))
    data_frame["sex_code"] = data_frame.sex.map(generate_column_mapping(data_frame['sex']))
    X = data_frame[['education_code','drugs_code','sex_code']]
    y = data_frame[["income"]]
    return X,y

def get_test_data_set(data_frame):
    X, y = get_data_set(data_frame)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2)
    return X_train, X_test, y_train, y_test


def test_performance(data_frame):
    X_train, X_test, y_train, y_test = get_test_data_set(data_frame)
    start = time();
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    train_score = mlr.score(X_train, y_train)
    test_score = mlr.score(X_test, y_test)
    print("execution time (sec) for test_performance:", time() - start)
    return train_score, test_score


def predict_income(data_frame, X_input):
    X, y = get_data_set(data_frame)
    start = time();
    mlr = LinearRegression()
    mlr.fit(X, y)
    prediction = mlr.predict(X_input)
    print("execution time (sec) for predict_income:", time() - start)
    return prediction, X_input

#Tests
df = pd.read_csv("profiles.csv")
print("What income has a person with 'graduated from law school' as education level, takes drugs often and is a woman:", predict_income(df, [[1,2,1]]))
print(test_performance(df))


#Results:
# ('execution time (sec) for predict_income:', 0.10599994659423828)
# ("What income has a person with 'graduated from law school' as education level, takes drugs often and is a woman:", (array([[13538.60577351]]), [[1, 2, 1]]))
# ('execution time (sec) for test_performance:', 0.007999897003173828)
# (0.008835104694037033, 0.00705630026514592)

#Visualize data
# x_train, x_test, y_train, y_test = get_test_data_set(df)
# mlr = LinearRegression()
# mlr.fit(x_train, y_train)
# y_predict = mlr.predict(x_test)
# plt.scatter(y_test, y_predict, alpha=0.4)
# plt.xlabel("Incomes")
# plt.ylabel("Predicted incomes")
# plt.title("Actual Income vs Predicted Income")
# plt.show()
# 
# plt.hist(df.education_code, bins=32)
# plt.xlabel("Education")
# plt.ylabel("Frequency")
# plt.xlim(0, 32)
# plt.show()

