import data.input as input
import pandas as pd

print(input.input_file_path())

data = input.structure(input.input_file_path())

print(data)	
