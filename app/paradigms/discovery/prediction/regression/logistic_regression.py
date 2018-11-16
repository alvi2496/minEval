import paradigms.discovery.prediction.trainer as trainer
from sklearn import linear_model

def accuracy(train_data_count, train_label, test_data_count, test_label):
	result = trainer.train(linear_model.LogisticRegression(), train_data_count, train_label, test_data_count, test_label)
	return result