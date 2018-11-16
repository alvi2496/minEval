import paradigms.discovery.prediction.classification.classifiers as classifiers
from sklearn import naive_bayes

def accuracy(train_data_count, train_label, test_data_count, test_label):
	result = classifiers.train(naive_bayes.MultinomialNB(), train_data_count, train_label, test_data_count, test_label)
	return result