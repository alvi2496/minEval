import paradigms.discovery.prediction.trainer as trainer
from sklearn import naive_bayes

def accuracy(train_data_count, train_label, test_data_count, test_label):
	result = trainer.train(naive_bayes.MultinomialNB(), train_data_count, train_label, test_data_count, test_label)
	return result