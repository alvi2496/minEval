from paradigms.verification.classification import decision_tree, naive_bayes, random_forest, svm
from paradigms.verification.regression import logistic_regression
from paradigms.verification import output, representer

def verify(data, test_size_for_split, k_fold_for_cross):

	result = output.initialize()

	result['split']['decision_tree']['count_vector'], result['split']['decision_tree']['tf_idf_vector'] = \
	decision_tree.split_validation(data)
	result['split']['naive_bayes']['count_vector'], result['split']['naive_bayes']['tf_idf_vector'] = \
	naive_bayes.split_validation(data)
	result['split']['random_forest']['count_vector'], result['split']['random_forest']['tf_idf_vector'] = \
	random_forest.split_validation(data)
	result['split']['svm']['count_vector'], result['split']['svm']['tf_idf_vector'] = \
	svm.split_validation(data)
	result['split']['logistic_regression']['count_vector'], result['split']['logistic_regression']['tf_idf_vector'] = \
	logistic_regression.split_validation(data)

	result['cross']['decision_tree']['count_vector']['array'], result['cross']['decision_tree']['count_vector']['mean'], \
	result['cross']['decision_tree']['tf_idf_vector']['array'], result['cross']['decision_tree']['tf_idf_vector']['mean'] = \
	decision_tree.cross_verification(data, k_fold_for_cross)
	result['cross']['naive_bayes']['count_vector']['array'], result['cross']['naive_bayes']['count_vector']['mean'], \
	result['cross']['naive_bayes']['tf_idf_vector']['array'], result['cross']['naive_bayes']['tf_idf_vector']['mean'] = \
	naive_bayes.cross_verification(data, k_fold_for_cross)
	result['cross']['random_forest']['count_vector']['array'], result['cross']['random_forest']['count_vector']['mean'], \
	result['cross']['random_forest']['tf_idf_vector']['array'], result['cross']['random_forest']['tf_idf_vector']['mean'] = \
	random_forest.cross_verification(data, k_fold_for_cross)
	result['cross']['svm']['count_vector']['array'], result['cross']['svm']['count_vector']['mean'], \
	result['cross']['svm']['tf_idf_vector']['array'], result['cross']['svm']['tf_idf_vector']['mean'] = \
	svm.cross_verification(data, k_fold_for_cross)
	result['cross']['logistic_regression']['count_vector']['array'], result['cross']['logistic_regression']['count_vector']['mean'], \
	result['cross']['logistic_regression']['tf_idf_vector']['array'], result['cross']['logistic_regression']['tf_idf_vector']['mean'] = \
	logistic_regression.cross_verification(data, k_fold_for_cross)

	representer.represent(result)
