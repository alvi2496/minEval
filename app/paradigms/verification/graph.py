import numpy as np
import matplotlib.pyplot as plt

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/3, 1.05*height, height, ha='center', va='bottom')        

def show(result):
	fig = plt.figure(1)

	algorithm_names = []
	feature_names = ['Count', 'TF-IDF', 'Word Embedd'] 
	for algo in result['algorithms']:
		algorithm_names.append(result['algorithms'][algo]['display_name'])	

	N = len(result['algorithms'])
	ind = np.arange(N) 
	width = .20

	ax_2 = fig.subplots()
	count_vectors = []
	for algo in result['algorithms']:
		count_vectors.append(round(result['algorithms'][algo]['verification_methods']['cross']['feature_vectors']['count']['value'], 2))       

	rects1 = ax_2.bar(ind, count_vectors, width, color='b')

	tf_idf_vectors = []
	for algo in result['algorithms']:
		tf_idf_vectors.append(round(result['algorithms'][algo]['verification_methods']['cross']['feature_vectors']['tf-idf']['value'], 2))

	rects2 = ax_2.bar(ind + width, tf_idf_vectors, width, color='g')

	word_vectors = []
	for algo in result['algorithms']:
		word_vectors.append(round(result['algorithms'][algo]['verification_methods']['cross']['feature_vectors']['word_embedd']['value'], 2))

	rects3 = ax_2.bar(ind + width + width, word_vectors, width, color='r')

	ax_2.set_ylabel('Accuracy')
	# ax_2.set_xlabel('Algotithms grouped by feature of data')
	ax_2.set_title('Accuracy using cross verification')
	ax_2.set_xticks(ind + width + width / 3 )
	ax_2.set_xticklabels(algorithm_names)

	ax_2.legend((rects1[0], rects2[0], rects3[0]), feature_names)

	autolabel(rects1, ax_2)
	autolabel(rects2, ax_2)
	autolabel(rects3, ax_2)

	plt.show()
