from paradigms.verification import graph
from prettytable import PrettyTable

def represent(result):
	table = PrettyTable()
	table.field_names = ['', 'Count', 'TF-IDF']
	for algo in result['algorithms']:
		row = [(result['algorithms'][algo]['display_name'])]
		for vt in result['algorithms'][algo]['verification_methods']['cross']['feature_vectors']:
			row.append(result['algorithms'][algo]['verification_methods']['cross']['feature_vectors'][vt]['value'])
		table.add_row(row)
	
	print(table)

	graph.show(result)
