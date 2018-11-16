import paradigms.discovery.prediction.trainer as trainer
from sklearn import ensemble

def accuracy(train_data_count, train_label, test_data_count, test_label):
	result = trainer.train(ensemble.RandomForestClassifier(), train_data_count, train_label, test_data_count, test_label)
	return result