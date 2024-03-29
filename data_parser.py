import pandas as pd
import types
import copy
import numpy as np
from sklearn.model_selection import train_test_split

# this class loads data from csv files into an object. Uses:
# (1) feed the data into a model during training
	# see get_data_splits
# (2) filter out variables you don't care about
	# see filter_variables
# (3) output csv files to files so that you can 
# create a folder with various variable combinations

# If not verbose, print statements indicating successful operations are suppressed.
# Print statements for failed operations are never suppressed, regardless of verbosity
class DataParser():

	# input_paths should be a list of paths to csv files
	def __init__(self, input_files, joinOn=['subNum'], verbose= False):
		temp_data_frames = []
		for input_file in input_files:
			try:
				if verbose:
					print('Loading data from file: "{}"'.format(input_file))
				temp_data_frames.append(pd.read_csv(input_file))
				if verbose:
					print('\t Done loading data for this file!\n')
			except ValueError as e:
				print('ValueError when trying to load data from file: "{}"'.format(input_file))
				print('Error message:', e)
				print('Continuing...')
				continue

		if len(temp_data_frames) == 0:
			raise RuntimeError('No data was successfully loaded :( Aborting')

		# after loading data frames, join frames using the keys (note this is an ordered op)
		# the frames are joined based on the keys in the order the keys are given. 
		# so frame1 is joined to frame2 using joinOn[0], then this frame to frame3 using joinOn[1], etc.
		self.data_frame = copy.deepcopy(temp_data_frames[0])
		for keys, frame in zip(joinOn, temp_data_frames[1:]):
			self.data_frame = pd.merge(self.data_frame, frame, on=keys)

		if verbose:
			print('Successfully joined dataframes by the following keys:', joinOn)

	def filter_data(self, key, value, verbose=False):
		self.data_frame = self.data_frame.loc[self.data_frame[key] == value]
		if verbose:
			print('Filtered data using key {} == {}'.format(key, value))

	def get_data_splits(self, train_vars, label_var=None, train_split=0.7, print_input_vars=True, verbose=True):
		# make sure no overlap in vars
		if label_var in train_vars:
			raise ValueError('Label variable {} found in train vars'.format(label_var))

		# make sure label_var is valid
		if label_var not in self.data_frame.columns:
			raise ValueError('Label variable {} not found in data'.format(label_var))

		# filter all non-train and non-label vars from dataframe
		full_name_train_vars = set()
		for var in train_vars:
			full_name_train_vars = full_name_train_vars.union(set(self.data_frame.filter(regex=var)))

		if len(full_name_train_vars) == 0:
			raise ValueError('No training variables found in data from list: {}'.format(train_vars))

		vars_to_keep = set([label_var]).union(full_name_train_vars)
		vars_to_drop = set(self.data_frame.columns).difference(vars_to_keep)

		# this data frame contains only the input variables and the label variable
		df = self.data_frame.drop(columns=vars_to_drop)

		# now convert to numpy matrices and validate the data
		train_matrix = df.drop(columns=label_var).values
		labels_matrix = df.filter(regex=label_var).values
		m_train, n = train_matrix.shape
		input_variables = list(df.drop(columns=label_var).columns)
		if (print_input_vars):
			print('Data has {} input variables:'.format(n))
			for i, var in enumerate(input_variables):
				print('\t ({}) {}'.format(i+1, var))
		m_labels, _ = labels_matrix.shape

		if verbose:
			print('\nOriginal # of rows in train matrix: {}'.format(m_train))
			print('Original # of rows in labels matrix: {}'.format(m_labels))
			print('Getting rid of invalid entries (like NaN)...')
		# first drop if label is nan
		valid_label_indices = ~np.isnan(labels_matrix).reshape((m_labels,))
		train_matrix = train_matrix[valid_label_indices,:]
		labels_matrix = labels_matrix[valid_label_indices]
		# then drop if any input features is nan
		valid_input_indices = ~np.isnan(train_matrix).any(axis=1)
		train_matrix = train_matrix[valid_input_indices,:]
		labels_matrix = labels_matrix[valid_input_indices]
		m_train, n = train_matrix.shape
		m_labels, _ = labels_matrix.shape
		if verbose:
			print('Cleaned # of rows in train matrix: {}'.format(m_train))
			print('Cleaned # of rows in labels matrix: {}\n'.format( m_labels))
		# Lastly, return train and test splits and the input var list
		X_train, X_test, y_train, y_test = train_test_split(train_matrix, labels_matrix, train_size=train_split)
		return (X_train, X_test, y_train, y_test, input_variables)


	def get_input_vars(self, input_vars):
		# this label_var doesn't matter -- as long as it exists, this hacky method will work
		label_var = 'rumination_type_score'
		X_train, X_test, _, _, input_variables = self.get_data_splits(train_vars=['Act'], label_var=label_var)
		return (np.concatenate((X_train, X_test)), input_variables)

	# pass in a single variable or list of variables to be filtered. returns 
	# a dataframe with these variables filtered out.
	# this will remove any variables that has a matching substring. For example, 
	# remove_vars('VAR1') will remove any variable name with 'VAR1' in it
	# def remove_vars(self, variables):
	# 	if not isinstance(variables, list):
	# 		variables = [variables]
	# 	frame = copy.deepcopy(self.data_frame)
	# 	all_removed_vars = []
	# 	for var in variables:
	# 		columns_to_drop = list(frame.filter(regex=var))
	# 		all_removed_vars += columns_to_drop
	# 		frame = frame[frame.columns.drop(columns_to_drop)]
	# 	if len(all_removed_vars) > 0:
	# 		print('Removed {} variables: {}'.format(len(all_removed_vars), all_removed_vars))
	# 	return frame

	# call this to output the csv to a specific file
	def save_csv_to_file(self, output_path=''):
		self.data_frame.to_csv(output_path)
