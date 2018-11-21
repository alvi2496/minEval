import data.processor as processor
# from paradigms.discovery.prediction.classification import naive_bayes, random_forest, svm
# from paradigms.discovery.prediction.regression import logistic_regression
import paradigms.verification.main as verifier
from sklearn import model_selection, preprocessing, metrics
from sklearn import decomposition

import xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import warnings

warnings.filterwarnings("ignore")

data = processor.structure(processor.input_file_path())

verifier.verify(data)
