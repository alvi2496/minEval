import data.processor as processor
from paradigms.discovery.prediction.classification import naive_bayes, random_forest, svm
from paradigms.discovery.prediction.regression import logistic_regression
from sklearn import model_selection, preprocessing, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition

import xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import warnings

warnings.filterwarnings("ignore")

data = processor.structure(processor.input_file_path())

train_data, test_data, train_label, test_label = processor.data_for_evaluation(data)

train_data_count, test_data_count = processor.count_vectorize(data, train_data, test_data)

# Accuracy in the respect of test data
print("Naive Bayes: ", naive_bayes.accuracy(train_data_count, train_label, test_data_count, test_label))
print("Logistic Regression: ", logistic_regression.accuracy(train_data_count, train_label, test_data_count, test_label))
print("Random Forest: ", random_forest.accuracy(train_data_count, train_label, test_data_count, test_label))
print("Support Vector Machine: ", svm.accuracy(train_data_count, train_label, test_data_count, test_label))