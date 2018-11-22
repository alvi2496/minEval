def initialize():
	result = {
		'split': {
			'naive_bayes': {
				'count_vector': 0,
				'tf_idf_vector': 0
			},
			'random_forest': {
				'count_vector': 0,
				'tf_idf_vector': 0
			},
			'svm': {
				'count_vector': 0,
				'tf_idf_vector': 0
			},
			'logistic_regression': {
				'count_vector': 0,
				'tf_idf_vector': 0
			}
		},
		'cross': {
			'naive_bayes': {
				'count_vector': {
					'array': [],
					'mean': 0,
				},
				'tf_idf_vector': {
					'array': [],
					'mean': 0
				}
			},
			'random_forest': {
				'count_vector': {
					'array': [],
					'mean': 0,
				},
				'tf_idf_vector': {
					'array': [],
					'mean': 0
				}
			},
			'svm': {
				'count_vector': {
					'array': [],
					'mean': 0,
				},
				'tf_idf_vector': {
					'array': [],
					'mean': 0
				}
			},
			'logistic_regression': {
				'count_vector': {
					'array': [],
					'mean': 0,
				},
				'tf_idf_vector': {
					'array': [],
					'mean': 0
				}
			}
		}
	}

	return result
	