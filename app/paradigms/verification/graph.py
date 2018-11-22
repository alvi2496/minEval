import numpy as np
import matplotlib.pyplot as plt

def autolabel(rects, ax):
	    for rect in rects:
	        height = rect.get_height()
	        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, height, ha='center', va='bottom')        

def show(result):
	N = len(result['algorithm_names'])

	ind = np.arange(N) 
	width = .20

	fig_1 = plt.figure() 
	ax_1 = fig_1.add_subplot(111)

	count_vectors = []
	for algo in result['split']:
		count_vectors.append(result['split'][algo]['count_vector'])       

	rects1 = ax_1.bar(ind, count_vectors, width, color='b')

	tf_idf_vectors = []
	for algo in result['split']:
		tf_idf_vectors.append(result['split'][algo]['tf_idf_vector'])

	rects2 = ax_1.bar(ind + width, tf_idf_vectors, width, color='g')

	ax_1.set_ylabel('Accuracy')
	ax_1.set_xlabel('Algotithms grouped by feature of data')
	ax_1.set_title('Accuracy using split verification')
	ax_1.set_xticks(ind + width / 2)
	ax_1.set_xticklabels(result['algorithm_names'])

	ax_1.legend((rects1[0], rects2[0]), result['feature_names'])

	autolabel(rects1, ax_1)
	autolabel(rects2, ax_1)
	
	fig_2 = plt.figure()
	ax_2 = fig_2.add_subplot(111)

	count_vectors = []
	for algo in result['cross']:
		count_vectors.append(round(result['cross'][algo]['count_vector']['mean'], 4))       

	rects1 = ax_2.bar(ind, count_vectors, width, color='b')

	tf_idf_vectors = []
	for algo in result['cross']:
		tf_idf_vectors.append(round(result['cross'][algo]['tf_idf_vector']['mean'], 4))

	rects2 = ax_2.bar(ind + width, tf_idf_vectors, width, color='g')

	ax_2.set_ylabel('Accuracy')
	ax_2.set_xlabel('Algotithms grouped by feature of data')
	ax_2.set_title('Accuracy using cross verification')
	ax_2.set_xticks(ind + width / 2)
	ax_2.set_xticklabels(result['algorithm_names'])

	ax_2.legend((rects1[0], rects2[0]), result['feature_names'])

	autolabel(rects1, ax_2)
	autolabel(rects2, ax_2)

	plt.show()
