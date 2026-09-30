import os
import pickle as pkl
import pandas as pd

def dataset_preprocess(df):
	
	return df

def read_dataset(file_path=None):
	if file_path.endswith(".pkl"):
		with open(file_path, "rb") as f:
			object = pkl.load(f,encoding='latin1')
		df = pd.DataFrame(object)
		return df
	elif file_path.endswith(".csv"):
		df = pd.read_csv(file_path)
		return df
	else:
		raise ValueError(f"Unsupported file type: {file_path}")