import pandas
from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def input_file_path(url):
	return url.strip()

def structure(data_file_path):
	data = pandas.read_csv(data_file_path, sep=",", header=None, names=['text', 'label'])
	return data

def data_for_evaluation(data):
	train_data, test_data, train_label, test_label = model_selection.train_test_split(data['text'], data['label'], test_size=0.4, random_state=0)
	return train_data, test_data, train_label, test_label

def count_vectorize(data, train_data, test_data):
	count_vector = CountVectorizer(analyzer = 'word', token_pattern = r'\w{1,}')
	count_vector.fit(data['text'])

	train_data_count = count_vector.transform(train_data)
	test_data_count = count_vector.transform(test_data)

	return train_data_count, test_data_count

def count_vectorizer(data):
	count_vect = CountVectorizer()
	train_data_count = count_vect.fit_transform(data['text'])
	
	return train_data_count

def tf_idf_vectorize(data, train_data, test_data):
	tf_idf_vector = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
	tf_idf_vector.fit(data['text'])
	train_data_tf_idf =  tf_idf_vector.transform(train_data)
	test_data_tf_idf =  tf_idf_vector.transform(test_data)

	return train_data_tf_idf, test_data_tf_idf

def tf_idf_vectorizer(data):
	tf_idf_vector = TfidfVectorizer()
	train_data_tf_idf = tf_idf_vector.fit_transform(data['text'])
	
	return train_data_tf_idf			

def encode_target_variable(train_label, test_label):
	encoder = preprocessing.LabelEncoder()
	train_label = encoder.fit_transform(train_label)
	test_label = encoder.fit_transform(test_label)
	return train_label, test_label		
