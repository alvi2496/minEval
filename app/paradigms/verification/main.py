from paradigms.verification.classification import naive_bayes, random_forest, svm
from paradigms.verification.regression import logistic_regression

def verify(data):

	print("Spliting dataset into train and test")
	print("Naive Bayes:", naive_bayes.split_validation(data))
	print("Random Forest: ", random_forest.split_validation(data))
	print("Support Vector Machine: ", svm.split_validation(data))
	print("Logistic Regression: ", logistic_regression.split_validation(data))
