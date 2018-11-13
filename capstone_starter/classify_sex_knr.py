#Predicts if "Is male" by job, education and income, using K-Nearest Neighbors Classifier model.

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from time import time
from sklearn import preprocessing

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
    columns = ["education","drugs","sex"]
    data_frame[columns] = data_frame[columns].replace(np.nan, '', regex=True)
    #Augment Data
    data_frame["education_code"] = data_frame.education.map(generate_column_mapping(data_frame["education"]))
    data_frame["drugs_code"] = data_frame.drugs.map(generate_column_mapping(data_frame['drugs']))
    X = data_frame[['education_code','drugs_code','income']]
    y = np.array(['m' == sex for sex in data_frame["sex"]])
    return X,y

def get_test_data_set(data_frame):
    X, y = get_data_set(data_frame)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2)
    return X_train, X_test, y_train, y_test


def test_performance(data_frame, iteration_count):
    X_train, X_test, y_train, y_test = get_test_data_set(data_frame)
    train_score = []
    test_score = []
    k_steps = []
    print("k iteration count", iteration_count)
    start = time();
    for k in range(1, iteration_count):
        classifier = KNeighborsClassifier(k, weights="distance")
        classifier.fit(X_train, y_train)
        train_score.append(classifier.score(X_train, y_train))
        test_score.append(classifier.score(X_test, y_test))
        k_steps.append(k)
    print("execution time (sec) for test_performance:", time() - start)
    plt.plot(k_steps, train_score, label="Training score")
    plt.plot(k_steps, test_score, label="Test score")    
    plt.legend(loc=8)
    plt.show()

def is_male(data_frame, x_input):
    X, y = get_data_set(data_frame)
    start = time();
    classifier = KNeighborsClassifier(7, weights="distance")
    classifier.fit(X, y)
    prediction = classifier.predict(x_input)[0],x_input
    print("execution time (sec) for is_male:", time() - start)
    return prediction

#Tests
df = pd.read_csv("profiles.csv")
print("Is male a person with 'graduated from law school' as education level, takes drugs often and has an income of 20000 $ per year:", is_male(df, [[1,2,20000]]))
test_performance(df, 100)

#Results:
# ('execution time (sec) for is_male:', 1.6140000820159912)
# ("Is male a person with 'graduated from law school' as education level, takes drugs often and has an income of 20000 $ per year:", (False, [[1, 2, 20000]]))
# ('k iteration count', 100)
# ('execution time (sec) for test_performance:', 502.60400009155273)