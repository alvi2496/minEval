from sklearn import metrics

def split_verifier(trained_classifier, test_data, test_label):
	predictions = trained_classifier.predict(test_data)
	return metrics.accuracy_score(predictions, test_label)