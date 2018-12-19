import re

ALGORITHMS = ['Decision Tree', 'Random Forest', 'Naive Bayes', 'SVM', 'Logistic Regression']
VERIFICATION_METHODS = ['Split', 'Cross']
FEATURE_VECTORS = ['Count', 'TF-IDF']


def ctk(value):
	key = value.lower()
	return re.sub(" ", "_", key)

def initialize():
	result = {
		'algorithms': {
		}
	}
	for a in ALGORITHMS:
		result['algorithms'].update([(ctk(a), {})])
		result['algorithms'][ctk(a)].update([('display_name', a), ('verification_methods', {} )])
		for b in VERIFICATION_METHODS:
			result['algorithms'][ctk(a)]['verification_methods'].update([(ctk(b), {})])
			result['algorithms'][ctk(a)]['verification_methods'][ctk(b)].update([('display_name', b),('feature_vectors', {})])
			for c in FEATURE_VECTORS:
				result['algorithms'][ctk(a)]['verification_methods'][ctk(b)]['feature_vectors'].update([(ctk(c), {})])
				result['algorithms'][ctk(a)]['verification_methods'][ctk(b)]['feature_vectors'][ctk(c)].update([('display_name', c), ('value', 0)])

	return result	
