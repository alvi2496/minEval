import numpy as np
import pandas
from nltk.corpus import stopwords
from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import scale
from keras.preprocessing import text
import pdb

STOPSET_WORDS = ['might', 'may', 'would', 'must', 'lgtm', 'could', 'can', 'good', 'great', 'nice', 'well', 'better', 'worse', \
	'worst', 'should', 'i', "i'll", "ill", "it's", "its", "im", "i'm", "they're", "theyre", "you're", "youre", "that's", 'btw', \
	"thats", "theres", "shouldnt", "shouldn't", "didn't", "didnt", "dont", "don't", "doesn't", "doesnt", "wasnt", "wasn't", \
	'sense', "mon", 'monday', 'tue', 'wed', 'wednesday', 'thursday', 'thu', 'friday', 'fri', 'sat', 'saturday', 'sun', 'sunday', \
	'jan', 'january', 'feb', 'february', 'mar', 'march', 'apr', 'april', 'may', 'jun', 'june', 'july', 'jul', 'aug', 'august',\
	'sep', 'september', 'oct', 'october', 'nov', 'novenber', 'dec', 'december', 'pm', 'am'    
]

WIKI_WORDS = 'app/data/files/wiki-news-300d-1M.vec'
encoding="utf-8"
WORD_DICTIONARY = {}
print('Loading word-embedding file and making dictionary...')
for line in open(WIKI_WORDS):
    values = line.split()
    WORD_DICTIONARY[values[0]] = np.array(values[1:], dtype='float32')

# token = text.Tokenizer()
# token.fit_on_texts(trainDF['text'])
# word_index = token.word_index

def normalize(np_array):
	np_array = scale(np_array, axis=0, with_mean=True, with_std=True, copy=True)
	min = np_array.min()
	max = np_array.max()
	if(max == 0 and min == 0):
		return np_array
	return (np_array - min) / (max - min)

def word_embed(data):
    # load the pre-trained word-embedding vectors 
    word_vector = np.zeros(np.array((data.shape[0], 300)))
    i = 0
    print('embedding word and converting to vector...')
    for sentence in data['text']:
        words = sentence.split()
        # length = len(words)
        for word in words:
                if word in WORD_DICTIONARY:
                        word_vector[i] = np.add(word_vector[i], WORD_DICTIONARY[word])
        word_vector[i] = normalize(word_vector[i])                
        i += 1
    return word_vector
    # token = text.Tokenizer()
    # token.fit_on_texts(data['text'])
    # word_index = token.word_index
    # # word_vector = sequence.pad_sequences(token.texts_to_sequences(data), maxlen=70)
    # embedding_matrix = np.zeros((len(word_index) + 1, 300))
    # for word, i in word_index.items():
    #     embedding_vector = WORD_DICTIONARY.get(word)
    #     if embedding_vector is not None:
    #         embedding_matrix[i] = embedding_vector
    # pdb.set_trace()        
    # return embedding_matrix

def input_file_path(url):
	return url.strip()

def structure(data_file_path):
	data = pandas.read_csv(data_file_path, sep=",", header=None, names=['text', 'label'])
	return data

def remove_stopwords(data):
	stopset = set(stopwords.words('english'))
	for word in STOPSET_WORDS:
		stopset.add(word) 
	
	data['text'] = data['text'].apply(lambda sentence: ' '.join([word for word in sentence.lower().split() if word not in (stopset)]))
	return data

def process(data_url):
	return remove_stopwords(structure(input_file_path(data_url)))		

def data_for_evaluation(data, test_size_for_split):
	train_data, test_data, train_label, test_label = model_selection.train_test_split(data['text'], data['label'], test_size=test_size_for_split, random_state=0)
	return train_data, test_data, train_label, test_label

def count_vectorize(data, train_data, test_data):
	count_vector = CountVectorizer(analyzer = 'word', token_pattern = r'\w{1,}')
	count_vector.fit(data['text'])

	train_data_count = count_vector.transform(train_data)
	test_data_count = count_vector.transform(test_data)

	# train_data_count = word_embed(train_data)
	# test_data_count = word_embed(test_data)

	# print('word embedding is complete!')

	return train_data_count, test_data_count

def count_vectorizer(data):
	count_vect = CountVectorizer()
	train_data_count = count_vect.fit_transform(data['text'])
	
	return train_data_count

def tf_idf_vectorize(data, train_data, test_data):
	tf_idf_vector = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range= (2,3), max_features=5000)
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
