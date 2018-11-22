import numpy as np
import matplotlib.pyplot as plt

def show(result):
	N = 4
	count_vectors = (
		result['split']['naive_bayes']['count_vector'],
		result['split']['random_forest']['count_vector'],
		result['split']['svm']['count_vector'],
		result['split']['logistic_regression']['count_vector']
	)
	men_std = (2, 3, 4, 1)

	ind = np.arange(N)  # the x locations for the groups
	width = 0.25       # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, count_vectors, width, color='b')

	tf_idf_vectors = (
		result['split']['naive_bayes']['tf_idf_vector'],
		result['split']['random_forest']['tf_idf_vector'],
		result['split']['svm']['tf_idf_vector'],
		result['split']['logistic_regression']['tf_idf_vector']
	)
	women_std = (3, 5, 2, 3)
	rects2 = ax.bar(ind + width, tf_idf_vectors, width, color='g')

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Accuracy')
	ax.set_xlabel('Algotithms grouped by feature of data')
	ax.set_title('Accuracy using split verification')
	ax.set_xticks(ind + width / 2)
	ax.set_xticklabels(('naive_bayes', 'random_forest', 'svm', 'logistic_regression'))

	ax.legend((rects1[0], rects2[0]), ('Split', 'Cross'))


	def autolabel(rects):
	    """
	    Attach a text label above each bar displaying its height
	    """
	    for rect in rects:
	        height = rect.get_height()
	        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
	                height,
	                ha='center', va='bottom')

	autolabel(rects1)
	autolabel(rects2)

	plt.show()