import paradigms.verification.trainer as trainer
import paradigms.verification.verifier as verifier
import data.processor as processor
from sklearn import naive_bayes

def split_validation(data, test_size_for_split):
	train_data, test_data, train_label, test_label = processor.data_for_evaluation(data, test_size_for_split)

	train_data_count, test_data_count = processor.count_vectorize(data, train_data, test_data)
	train_data_tf_idf, test_data_tf_idf = processor.tf_idf_vectorize(data, train_data, test_data)

	trained_classifier_with_count = trainer.train(naive_bayes.MultinomialNB(), train_data_count, train_label)
	trained_classifier_with_tf_idf = trainer.train(naive_bayes.MultinomialNB(), train_data_tf_idf, train_label)

	result_with_count = verifier.split_verifier(trained_classifier_with_count, test_data_count, test_label)
	result_with_tf_idf = verifier.split_verifier(trained_classifier_with_tf_idf, test_data_tf_idf, test_label)

	return result_with_count, result_with_tf_idf

def cross_verification(data, feature_vectors, n):
	count_resut_array, count_result = verifier.cross_verifier(naive_bayes.MultinomialNB(), \
		feature_vectors['count'], data['label'], n)
	tf_idf_result_array, tf_idf_result = verifier.cross_verifier(naive_bayes.MultinomialNB(), \
		feature_vectors['tf-idf'], data['label'], n)
	return count_result, tf_idf_result   
	