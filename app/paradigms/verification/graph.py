import numpy as np
import matplotlib.pyplot as plt

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, height, ha='center', va='bottom')        

def show(result):
	fig = plt.figure(1)

	algorithm_names = []
	feature_names = ['Count', 'TF-IDF'] 
	for algo in result['algorithms']:
		algorithm_names.append(result['algorithms'][algo]['display_name'])	

	N = len(result['algorithms'])
	ind = np.arange(N) 
	width = .20

	# ax_1 = fig.add_subplot(211)

	# count_vectors = []
	# for algo in result['algorithms']:
	# 	count_vectors.append(result['algorithms'][algo]['verification_methods']['split']['feature_vectors']['count']['value'])       

	# rects1 = ax_1.bar(ind, count_vectors, width, color='b')

	# tf_idf_vectors = []
	# for algo in result['algorithms']:
	# 	tf_idf_vectors.append(result['algorithms'][algo]['verification_methods']['split']['feature_vectors']['tf-idf']['value'])

	# rects2 = ax_1.bar(ind + width, tf_idf_vectors, width, color='g')

	# ax_1.set_ylabel('Accuracy')
	# # ax_1.set_xlabel('Algotithms grouped by feature of data')
	# ax_1.set_title('Accuracy using split verification')
	# ax_1.set_xticks(ind + width / 2)
	# ax_1.set_xticklabels(algorithm_names)

	# ax_1.legend((rects1[0], rects2[0]), feature_names)

	# autolabel(rects1, ax_1)
	# autolabel(rects2, ax_1)

	count_vectors = []
	for algo in result['algorithms']:
		count_vectors.append(round(result['algorithms'][algo]['verification_methods']['cross']['feature_vectors']['count']['value'], 4))       

	rects1 = fig.bar(ind, count_vectors, width, color='b')

	tf_idf_vectors = []
	for algo in result['algorithms']:
		tf_idf_vectors.append(round(result['algorithms'][algo]['verification_methods']['cross']['feature_vectors']['tf-idf']['value'], 4))

	rects2 = fig.bar(ind + width, tf_idf_vectors, width, color='g')

	fig.set_ylabel('Accuracy')
	# ax_2.set_xlabel('Algotithms grouped by feature of data')
	fig.set_title('Accuracy using cross verification')
	fig.set_xticks(ind + width / 2)
	fig.set_xticklabels(algorithm_names)

	fig.legend((rects1[0], rects2[0]), feature_names)

	# autolabel(rects1, ax_2)
	autolabel(rects2, fig)

	plt.show()
