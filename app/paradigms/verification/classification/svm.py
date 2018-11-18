import paradigms.discovery.prediction.trainer as trainer
from sklearn import svm

def accuracy(train_data_count, train_label, test_data_count, test_label):
	result = trainer.train(svm.SVC(), train_data_count, train_label, test_data_count, test_label)
	return result

import paradigms.verification.trainer as trainer
import paradigms.verification.verifier as verifier
import data.processor as processor
from sklearn import svm

def split_validation(data):
	train_data, test_data, train_label, test_label = processor.data_for_evaluation(data)

	train_data_count, test_data_count = processor.count_vectorize(data, train_data, test_data)

	trained_classifier = trainer.train(svm.SVC(), train_data_count, train_label)

	result = verifier.split_verifier(trained_classifier, test_data_count, test_label)

	return result	