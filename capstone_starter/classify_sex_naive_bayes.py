#Predicts if "Is male" by word counts of:
#    essay3 - The first thing people usually notice about me
#    essay6 - I spend a lot of time thinking about
#    essay8 - The most private thing I am willing to admit
#    essay0 - My self summary. 
# The prediction model is MultinomialNB (Naive Bayes Classifier).

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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
    X_columns = ["essay6","essay8","essay0","essay3"]
    data_frame[X_columns] = data_frame[X_columns].replace(np.nan, '', regex=True)
    data_frame["sex"] = data_frame["sex"].replace(np.nan, '', regex=True)
    male_words_list,not_male_words_list = get_words_lists(data_frame, X_columns, "sex")
    counter = CountVectorizer()
    counter.fit(not_male_words_list + male_words_list)
    X = counter.transform(not_male_words_list + male_words_list)
    y = [0] * len(not_male_words_list) + [1] * len(male_words_list)
    return X, y, counter


def get_test_data_set(data_frame):
    X, y, counter = get_data_set(data_frame)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2)
    return X_train, X_test, y_train, y_test


def test_performance(data_frame):
    X_train, X_test, y_train, y_test = get_test_data_set(data_frame)
    start = time()
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    y_predicted = classifier.predict(X_test)
    print(classifier.score(X_test, y_test))
    result = confusion_matrix(y_test,y_predicted)
    accuracy_score_result = accuracy_score(y_test,y_predicted)
    precision_result = precision_score(y_test,y_predicted)
    recall_result = recall_score(y_test,y_predicted)
    print("execution time (sec) for test_performance:", time() - start)
    return result,accuracy_score_result,precision_result,recall_result
    

def is_male(data_frame, review):
    X, y, counter = get_data_set(data_frame)
    start = time()
    review_counts = counter.transform([review])
    classifier = MultinomialNB()
    classifier.fit(X, y)
    prediction = classifier.predict(review_counts), classifier.predict_proba(review_counts),review
    print("execution time (sec) for is_male:", time() - start)
    return prediction

#Tests
df = pd.read_csv("profiles.csv")
# print(is_male(df, "i want gud male"))
# print(is_male(df, "i'm from rich city"))
# print(is_male(df, "female friends only"))
print(is_male(df, "i believe that life is a bold, dashing adventure"))
print(test_performance(df))

# for index, row in df.iterrows():
#      print(row["sex"],row["essay0"])

#Results:
# ('execution time (sec) for is_male:', 0.046000003814697266)
# (array([1]), array([[0.39979767, 0.60020233]]), 'i want gud male')
# ('execution time (sec) for is_male:', 0.0409998893737793)
# (array([1]), array([[0.39979767, 0.60020233]]), "i'm from rich city")
# ('execution time (sec) for is_male:', 0.0409998893737793)
# (array([1]), array([[0.39979767, 0.60020233]]), 'female friends only')
# ('execution time (sec) for is_male:', 0.0409998893737793)
# (array([1]), array([[0.03029627, 0.96970373]]), 'i believe that life is a bold, dashing adventure')
# 0.5998372890948321
# ('execution time (sec) for test_performance:', 0.06999993324279785)
# [[  154 11170]
#  [  143 16804]]