import numpy as np
import matplotlib.pyplot as plt

def show(result):
	N = len(result['algorithm_names'])
	count_vectors = []
	for algo in result['split']:
		count_vectors.append(result['split'][algo]['count_vector'])
	
	ind = np.arange(N) # the x locations for the groups
	width = .20       # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, count_vectors, width, color='b')

	tf_idf_vectors = []
	for algo in result['split']:
		tf_idf_vectors.append(result['split'][algo]['tf_idf_vector'])

	rects2 = ax.bar(ind + width, tf_idf_vectors, width, color='g')

	ax.set_ylabel('Accuracy')
	ax.set_xlabel('Algotithms grouped by feature of data')
	ax.set_title('Accuracy using split verification')
	ax.set_xticks(ind + width / 2)
	ax.set_xticklabels(result['algorithm_names'])

	ax.legend((rects1[0], rects2[0]), result['feature_names'])

	def autolabel(rects):
	    for rect in rects:
	        height = rect.get_height()
	        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, height, ha='center', va='bottom')

	autolabel(rects1)
	autolabel(rects2)

	plt.show()
