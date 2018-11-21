from sklearn import metrics
from sklearn.model_selection import cross_val_score

def split_verifier(trained_classifier, test_data, test_label):
	predictions = trained_classifier.predict(test_data)
	return metrics.accuracy_score(predictions, test_label)

def cross_verifier(classifier, text, label, n):
	result_array = cross_val_score(classifier, text, label, cv=n)
	result = sum(result_array) / n	
	return result_array, result	