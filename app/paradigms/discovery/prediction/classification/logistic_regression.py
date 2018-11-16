import paradigms.discovery.prediction.classification.classifiers as classifiers
from sklearn import linear_model

def accuracy(train_data_count, train_label, test_data_count, test_label):
	result = classifiers.train(linear_model.LogisticRegression(), train_data_count, train_label, test_data_count, test_label)
	return result