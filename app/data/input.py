import pandas as pd

def input_file_path():
	# data_file_path = input("Path to data file: ")
	data_file_path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSGhxSbBeeXdkRdBVQ9wSL1aJTs52SXV3NKfcfoX1wI89XDCJMC5tW0HZk5HYdh2xT0DtufMLSn9hHX/pub?gid=1193567183&single=true&output=csv"
	data_file_path.strip()
	return data_file_path

def structure(data_file_path):
	data = pd.read_csv(data_file_path, sep=",", header=None, names=['text', 'value'])
	return data