from paradigms.verification.classification import decision_tree, naive_bayes, random_forest, svm
from paradigms.verification.regression import logistic_regression
from paradigms.verification import output, representer

def verify(data, test_size_for_split, k_fold_for_cross):

	result = output.initialize()

	# Split Verification
	result['algorithms']['decision_tree']['verification_methods']['split']['feature_vectors']['count']['value'], \
	result['algorithms']['decision_tree']['verification_methods']['split']['feature_vectors']['tf-idf']['value'] = \
	decision_tree.split_validation(data, test_size_for_split)

	result['algorithms']['random_forest']['verification_methods']['split']['feature_vectors']['count']['value'], \
	result['algorithms']['random_forest']['verification_methods']['split']['feature_vectors']['tf-idf']['value'] = \
	random_forest.split_validation(data, test_size_for_split)

	result['algorithms']['naive_bayes']['verification_methods']['split']['feature_vectors']['count']['value'], \
	result['algorithms']['naive_bayes']['verification_methods']['split']['feature_vectors']['tf-idf']['value'] = \
	naive_bayes.split_validation(data, test_size_for_split)

	result['algorithms']['svm']['verification_methods']['split']['feature_vectors']['count']['value'], \
	result['algorithms']['svm']['verification_methods']['split']['feature_vectors']['tf-idf']['value'] = \
	svm.split_validation(data, test_size_for_split)

	result['algorithms']['logistic_regression']['verification_methods']['split']['feature_vectors']['count']['value'], \
	result['algorithms']['logistic_regression']['verification_methods']['split']['feature_vectors']['tf-idf']['value'] = \
	logistic_regression.split_validation(data, test_size_for_split)

	#Cross Verification
	result['algorithms']['decision_tree']['verification_methods']['cross']['feature_vectors']['count']['value'], \
	result['algorithms']['decision_tree']['verification_methods']['cross']['feature_vectors']['tf-idf']['value'] = \
	decision_tree.cross_verification(data, k_fold_for_cross)

	result['algorithms']['random_forest']['verification_methods']['cross']['feature_vectors']['count']['value'], \
	result['algorithms']['random_forest']['verification_methods']['cross']['feature_vectors']['tf-idf']['value'] = \
	random_forest.cross_verification(data, k_fold_for_cross)

	result['algorithms']['naive_bayes']['verification_methods']['cross']['feature_vectors']['count']['value'], \
	result['algorithms']['naive_bayes']['verification_methods']['cross']['feature_vectors']['tf-idf']['value'] = \
	naive_bayes.cross_verification(data, k_fold_for_cross)

	result['algorithms']['svm']['verification_methods']['cross']['feature_vectors']['count']['value'], \
	result['algorithms']['svm']['verification_methods']['cross']['feature_vectors']['tf-idf']['value'] = \
	svm.cross_verification(data, k_fold_for_cross)

	result['algorithms']['logistic_regression']['verification_methods']['cross']['feature_vectors']['count']['value'], \
	result['algorithms']['logistic_regression']['verification_methods']['cross']['feature_vectors']['tf-idf']['value'] = \
	logistic_regression.cross_verification(data, k_fold_for_cross)

	representer.represent(result)
