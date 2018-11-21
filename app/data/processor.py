import pandas
from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import CountVectorizer

def input_file_path():
	# data_file_path = input("Path to data file: ")
	data_file_path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSGhxSbBeeXdkRdBVQ9wSL1aJTs52SXV3NKfcfoX1wI89XDCJMC5tW0HZk5HYdh2xT0DtufMLSn9hHX/pub?gid=1193567183&single=true&output=csv"
	data_file_path.strip()
	return data_file_path

def structure(data_file_path):
	data = pandas.read_csv(data_file_path, sep=",", header=None, names=['text', 'label'])
	return data

def data_for_evaluation(data):
	train_data, test_data, train_label, test_label = model_selection.train_test_split(data['text'], data['label'], test_size=0.4, random_state=0)
	return train_data, test_data, train_label, test_label

def count_vectorize(data, train_data, test_data):
	# Create a count vectorizer object
	count_vector = CountVectorizer(analyzer = 'word', token_pattern = r'\w{1,}')
	count_vector.fit(data['text'])

	# Transform the data
	train_data_count = count_vector.transform(train_data)
	test_data_count = count_vector.transform(test_data)

	return train_data_count, test_data_count

def count_vectorizer(data):
	count_vect = CountVectorizer()
	train_data_count = count_vect.fit_transform(data['text'])
	# train_label_count = encoder.fit_transform(data.label)
	return train_data_count	

def encode_target_variable(train_label, test_label):
	encoder = preprocessing.LabelEncoder()
	train_label = encoder.fit_transform(train_label)
	test_label = encoder.fit_transform(test_label)
	return train_label, test_label		
