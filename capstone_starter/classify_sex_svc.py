#Predicts if "Is male" by word counts of:
#    essay3 - The first thing people usually notice about me
#    essay6 - I spend a lot of time thinking about
#    essay8 - The most private thing I am willing to admit
#    essay0 - My self summary. 
# The prediction model is SVC(Support Vector Machines).

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from time import time

def get_col_idx(data_frame, column_name):
    return data_frame.columns.get_loc(column_name)

def get_word(df_word):
    word = ''
    for char in df_word:
        if char.isalnum():
            word += char
    return word

def get_words_lists(data_frame, X_columns, label_column):
    male_words_list = []
    not_male_words_list = []
    for index, row in data_frame.iterrows():
        if ('m' == row[label_column]):
            for column_name in X_columns:
                if (type(row[column_name]) == str and len(row[column_name]) > 0):
                    male_words_list.append(get_word(row[column_name]))
        else:
            for column_name in X_columns:
                if (type(row[column_name])  == str and len(row[column_name]) > 0):
                    not_male_words_list.append(get_word(row[column_name]))

    return male_words_list,not_male_words_list

def get_data_set(data_frame):
    X_columns = ["essay6","essay8","essay0"]
    data_frame[X_columns] = data_frame[X_columns].replace(np.nan, '', regex=True)
    data_frame["sex"] = data_frame["sex"].replace(np.nan, '', regex=True)
    male_words_list,not_male_words_list = get_words_lists(data_frame, X_columns, "sex")
    counter = CountVectorizer()
    counter.fit(not_male_words_list + male_words_list)
    X = counter.transform(not_male_words_list + male_words_list)
    y = [0] * len(not_male_words_list) + [1] * len(male_words_list)
    return X, y, counter

def is_male(data_frame, review):
    X, y, counter = get_data_set(data_frame)
    review_counts = counter.transform([review])
    start = time()
    classifier = SVC(kernel='linear', C = 0.09)
    #classifier = LinearSVC()
    classifier.fit(X, y)
    prediction = classifier.predict(review_counts)
    print("execution time (sec) for is_male:", time() - start)
    return prediction, review


def get_test_data_set(data_frame):
    X, y, counter = get_data_set(data_frame)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2)
    return X_train, X_test, y_train, y_test


def test_performance(data_frame):
    X_train, X_test, y_train, y_test = get_test_data_set(data_frame)
    start = time()
    classifier = SVC(kernel='linear', C = 0.01)
    #classifier = LinearSVC()
    classifier.fit(X_train, y_train)
    y_predicted = classifier.predict(X_test)
    accuracy_score_result = accuracy_score(y_test,y_predicted)
    precision_result = precision_score(y_test,y_predicted)
    recall_result = recall_score(y_test,y_predicted)
    print("execution time (sec) for test_performance:", time() - start)
    return confusion_matrix(y_test,y_predicted),accuracy_score_result,precision_result,recall_result

#Tests
df = pd.read_csv("profiles.csv")
# print(is_male(df, "i want gud male"))
# print(is_male(df, "i'm from rich city"))
# print(is_male(df, "female friends only"))
print(is_male(df, "i believe that life is a bold, dashing adventure"))
print(test_performance(df))

#Results:
# ('execution time (sec) for is_male:', 896.4509999752045)
# (array([1]), 'i want gud male')
# ('execution time (sec) for is_male:', 898.3940000534058)
# (array([1]), "i'm from rich city")
# ('execution time (sec) for is_male:', 959.7420001029968)
# (array([1]), 'female friends only')
# ('execution time (sec) for is_male:', 971.396999835968)
# (array([1]), 'i believe that life is a bold, dashing adventure')
# 0.6021010929928194
# ('execution time (sec) for test_performance:', 623.7509999275208)
# [[    0 11249]
#  [    0 17022]]

# for index, row in df.iterrows():
#      print(row["sex"],row["essay0"])
