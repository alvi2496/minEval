import paradigms.verification.trainer as trainer
import paradigms.verification.verifier as verifier
import data.processor as processor
from sklearn import linear_model

def split_validation(data):
	train_data, test_data, train_label, test_label = processor.data_for_evaluation(data)

	train_data_count, test_data_count = processor.count_vectorize(data, train_data, test_data)

	trained_classifier = trainer.train(linear_model.LogisticRegression(), train_data_count, train_label)

	result = verifier.split_verifier(trained_classifier, test_data_count, test_label)

	return result

def cross_verification(data, n):
	return verifier.cross_verifier(linear_model.LogisticRegression(), processor.count_vectorizer(data), data['label'], n) 		