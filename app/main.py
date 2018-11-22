import data.processor as processor
import paradigms.verification.controller as verifier
from sklearn import model_selection, preprocessing, metrics
from sklearn import decomposition

import xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import warnings

warnings.filterwarnings("ignore")

data_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSGhxSbBeeXdkRdBVQ9wSL1aJTs52SXV3NKfcfoX1wI89XDCJMC5tW0HZk5HYdh2xT0DtufMLSn9hHX/pub?gid=1193567183&single=true&output=csv"

data = processor.structure(processor.input_file_path(data_url))

verifier.verify(data)
