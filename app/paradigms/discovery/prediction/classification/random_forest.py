import paradigms.discovery.prediction.classification.classifiers as classifiers
from sklearn import ensemble

def accuracy(train_data_count, train_label, test_data_count, test_label):
	result = classifiers.train(ensemble.RandomForestClassifier(), train_data_count, train_label, test_data_count, test_label)
	return result