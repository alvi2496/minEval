import data.input as input
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

print(input.input_file_path())

data = input.structure(input.input_file_path())

# print(data)	

train_data, test_data, train_label, test_label = input.data_for_evaluation(data)

print(train_data)
# print(train_label)
print(test_data)
# print(test_label)

train_data_count, test_data_count = input.count_vectorize(data, train_data, test_data)

print(train_data_count)

# print(test_data_count)
