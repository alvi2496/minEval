import paradigms.discovery.prediction.trainer as trainer
from sklearn import svm

def accuracy(train_data_count, train_label, test_data_count, test_label):
	result = trainer.train(svm.SVC(), train_data_count, train_label, test_data_count, test_label)
	return result