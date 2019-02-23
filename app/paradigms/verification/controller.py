from paradigms.verification.classification import decision_tree, naive_bayes, random_forest, svm
from paradigms.verification.regression import logistic_regression
from paradigms.verification import output, representer
import data.processor as processor

def verify(data, folds):

	result = output.initialize()

	feature_vectors = {}
	feature_vectors['count'] = processor.count_vectorizer(data)
	feature_vectors['tf-idf'] = processor.tf_idf_vectorizer(data)
	feature_vectors['word-embedd'] = processor.word_embed(data)
	
	#Cross Verification
	result['algorithms']['decision_tree']['verification_methods']['cross']['feature_vectors']['count']['value'], \
	result['algorithms']['decision_tree']['verification_methods']['cross']['feature_vectors']['tf-idf']['value'], \
	result['algorithms']['decision_tree']['verification_methods']['cross']['feature_vectors']['word_embedd']['value'] = \
	decision_tree.cross_verification(data, feature_vectors, folds)

	result['algorithms']['random_forest']['verification_methods']['cross']['feature_vectors']['count']['value'], \
	result['algorithms']['random_forest']['verification_methods']['cross']['feature_vectors']['tf-idf']['value'], \
	result['algorithms']['random_forest']['verification_methods']['cross']['feature_vectors']['word_embedd']['value'] = \
	random_forest.cross_verification(data, feature_vectors, folds)

	result['algorithms']['naive_bayes']['verification_methods']['cross']['feature_vectors']['count']['value'], \
	result['algorithms']['naive_bayes']['verification_methods']['cross']['feature_vectors']['tf-idf']['value'] = \
	naive_bayes.cross_verification(data, feature_vectors, folds)

	result['algorithms']['svm']['verification_methods']['cross']['feature_vectors']['count']['value'], \
	result['algorithms']['svm']['verification_methods']['cross']['feature_vectors']['tf-idf']['value'], \
	result['algorithms']['svm']['verification_methods']['cross']['feature_vectors']['word_embedd']['value'] = \
	svm.cross_verification(data, feature_vectors, folds)

	result['algorithms']['logistic_regression']['verification_methods']['cross']['feature_vectors']['count']['value'], \
	result['algorithms']['logistic_regression']['verification_methods']['cross']['feature_vectors']['tf-idf']['value'], \
	result['algorithms']['logistic_regression']['verification_methods']['cross']['feature_vectors']['word_embedd']['value'] = \
	logistic_regression.cross_verification(data, feature_vectors, folds)

	representer.represent(result)
