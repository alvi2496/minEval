from paradigms.verification import graph

def represent(result):
	print("---------------------------------------------------")
	print("Spliting dataset into train and test")
	print("***************************************************")
	for algo in result['algorithms']:
		print(result['algorithms'][algo]['display_name'])
		for vt in result['algorithms'][algo]['verification_methods']['split']['feature_vectors']:
			print("With", result['algorithms'][algo]['verification_methods']['split']['feature_vectors'][vt]['display_name'], ":", \
			 result['algorithms'][algo]['verification_methods']['split']['feature_vectors'][vt]['value'])
		print("###################################################")
	print("---------------------------------------------------")		

	print("---------------------------------------------------")
	print("Cross validation")
	print("***************************************************")
	for algo in result['algorithms']:
		print(result['algorithms'][algo]['display_name'])
		for vt in result['algorithms'][algo]['verification_methods']['cross']['feature_vectors']:
			print("With", result['algorithms'][algo]['verification_methods']['cross']['feature_vectors'][vt]['display_name'], ":", \
			 result['algorithms'][algo]['verification_methods']['cross']['feature_vectors'][vt]['value'])
		print("###################################################")
	print("---------------------------------------------------")

	graph.show(result)
