import data.input as input
from paradigms.discovery.prediction.classification import naive_bayes, logistic_regression, random_forest
from sklearn import model_selection, preprocessing, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition

import xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

data = input.structure(input.input_file_path())

train_data, test_data, train_label, test_label = input.data_for_evaluation(data)

train_data_count, test_data_count = input.count_vectorize(data, train_data, test_data)

# Accuracy in the respect of test data
print("Naive Bayes: ", naive_bayes.accuracy(train_data_count, train_label, test_data_count, test_label))
print("Logistic Regression: ", logistic_regression.accuracy(train_data_count, train_label, test_data_count, test_label))
print("Random Forest: ", random_forest.accuracy(train_data_count, train_label, test_data_count, test_label))