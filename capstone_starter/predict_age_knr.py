# Predicts age with income and job, using K-Nearest Neighbors Regression model.

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
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
    data_frame["job"] = data_frame["job"].replace(np.nan, '', regex=True)
    data_frame["income"] = data_frame["income"].replace(np.nan, -1, regex=True)
    data_frame["age"] = data_frame["age"].replace(np.nan, -1, regex=True)
    #Augment Data
    data_frame["job_code"] = data_frame.job.map(generate_column_mapping(data_frame["job"]))
    x = data_frame[['income','job_code']]
    y = data_frame[["age"]]
    return x,y

def get_test_data_set(data_frame):
    x, y = get_data_set(data_frame)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)
    return x_train, x_test, y_train, y_test


def test_performance(data_frame, iteration_count):
    x_train, x_test, y_train, y_test = get_test_data_set(data_frame)
    train_score = []
    test_score = []
    k_steps = []
    start = time();
    for k in range(1, iteration_count):
        regressor = KNeighborsRegressor(k, weights="distance")
        regressor.fit(x_train, y_train)
        train_score.append(regressor.score(x_train, y_train))
        test_score.append(regressor.score(x_test, y_test))
        k_steps.append(k)
    print("execution time (sec) for test_performance:", time() - start)
    plt.plot(k_steps, train_score, label="Training score")
    plt.plot(k_steps, test_score, label="Test score")   
    plt.xlabel('k values')
    plt.ylabel('Accuracy score')
    plt.legend(loc=8)
    plt.show()


def predict_age(data_frame, x_input):
    x, y = get_data_set(data_frame)
    start = time();
    regressor = KNeighborsRegressor(17, weights="distance")
    regressor.fit(x, y)
    prediction = regressor.predict(x_input)
    print("execution time (sec) for predict_income:", time() - start)
    return prediction

#Tests
df = pd.read_csv("profiles.csv")
print("What age is a person earning 50000 $ per year and job as 'science / tech / engineering':", predict_age(df, [[50000, 6]]))
test_performance(df, 100)


#Visualize data
# x_train, x_test, y_train, y_test = get_test_data_set(df)
# regressor = KNeighborsRegressor(17, weights="distance")
# regressor.fit(x_train, y_train)
# y_predict = regressor.predict(x_test)
# plt.scatter(y_test, y_predict, alpha=0.4)
# plt.xlabel("Ages")
# plt.ylabel("Predicted ages")
# plt.title("Actual Age vs Predicted Age")
# plt.show()
#

