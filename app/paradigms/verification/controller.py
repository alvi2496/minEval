from paradigms.verification.classification import decision_tree, naive_bayes, random_forest, svm
from paradigms.verification.regression import logistic_regression
from paradigms.verification import output, representer
import data.processor as processor

def verify(data, folds):

	result = output.initialize()

	os_data = {}

	feature_vectors = {}
	feature_vectors['count'], os_data['label'] = processor.oversample(processor.count_vectorizer(data), data['label'])
	feature_vectors['tf-idf'], os_data['label'] = processor.oversample(processor.tf_idf_vectorizer(data), data['label'])
	feature_vectors['word-embedd'], os_data['label'] = processor.oversample(processor.word_embed(data), data['label'])
	
	#Cross Verification
	result['algorithms']['decision_tree']['verification_methods']['cross']['feature_vectors']['count']['value'], \
	result['algorithms']['decision_tree']['verification_methods']['cross']['feature_vectors']['tf-idf']['value'], \
	result['algorithms']['decision_tree']['verification_methods']['cross']['feature_vectors']['word_embedd']['value'] = \
	decision_tree.cross_verification(os_data, feature_vectors, folds)

	result['algorithms']['random_forest']['verification_methods']['cross']['feature_vectors']['count']['value'], \
	result['algorithms']['random_forest']['verification_methods']['cross']['feature_vectors']['tf-idf']['value'], \
	result['algorithms']['random_forest']['verification_methods']['cross']['feature_vectors']['word_embedd']['value'] = \
	random_forest.cross_verification(os_data, feature_vectors, folds)

	result['algorithms']['naive_bayes']['verification_methods']['cross']['feature_vectors']['count']['value'], \
	result['algorithms']['naive_bayes']['verification_methods']['cross']['feature_vectors']['tf-idf']['value'] = \
	naive_bayes.cross_verification(os_data, feature_vectors, folds)

	result['algorithms']['svm']['verification_methods']['cross']['feature_vectors']['count']['value'], \
	result['algorithms']['svm']['verification_methods']['cross']['feature_vectors']['tf-idf']['value'], \
	result['algorithms']['svm']['verification_methods']['cross']['feature_vectors']['word_embedd']['value'] = \
	svm.cross_verification(os_data, feature_vectors, folds)

	result['algorithms']['logistic_regression']['verification_methods']['cross']['feature_vectors']['count']['value'], \
	result['algorithms']['logistic_regression']['verification_methods']['cross']['feature_vectors']['tf-idf']['value'], \
	result['algorithms']['logistic_regression']['verification_methods']['cross']['feature_vectors']['word_embedd']['value'] = \
	logistic_regression.cross_verification(os_data, feature_vectors, folds)

	representer.represent(result)
