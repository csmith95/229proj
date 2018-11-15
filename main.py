import matplotlib.pyplot as plt
from data_parser import *
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import argparse
from sklearn.cluster import KMeans
from tabulate import tabulate
import hypertools as hype

ALL_BIOTYPES = ['rumination', 'anxious_avoid', 'negative_bias', 'con_threat_dysreg', \
					'noncon_threat_dysreg', 'anhedonia', 'cog_dyscontrol', 'inattention'] 

# execution starts here
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train a model.')
	parser.add_argument('--model', type=str, help='type of model to train/evaluate', required=True)
	parser.add_argument('--n_clusters', type=int, help='number of clusters for k-means')
	parser.add_argument('--plot', type=bool)
	# options for prediction task: 
		# Linear Regression Tasks
			# (i) 'biotype' --> fits model to predict biotype scores from fMRI
			# (ii) 'emotion' --> fits model to predict average emotional recognition accuracy using fMRI
		# Clustering Tasks
			# None (just breaks data into n_clusters)
	parser.add_argument('--task', type=str, help='prediction task')
	parser.add_argument('--biotypes', type=str, help='reduced biotypes to consider as labels \
													when fitting linear regression',
												nargs='+', default=' '.join(ALL_BIOTYPES))
	parser.add_argument('--show_coeffs', type=bool, help='show coefficients for linear reg', \
															default=False)

	args = parser.parse_args()
	print()
	if args.model == 'linreg':
		print('Training a Linear Regression Model...')
		if args.task == 'biotype':
			dataParser = DataParser(input_files=['./data/Neuroimaging/Reduced_biotype_imaging_data_s1.csv'])
			all_MSE = []
			for i, biotype in enumerate(args.biotypes.split()):
				print()
				biotype_variable_name = '{}_type_score'.format(biotype)
				X_train, X_test, y_train, y_test, input_variables = dataParser.get_data_splits(train_vars=['Act', 'PPI'], \
															label_var=biotype_variable_name, print_input_vars=(i==0))
				model = linear_model.LinearRegression()
				model.fit(X_train, y_train)
				if (args.show_coeffs):
					coefficient_list = model.coef_.tolist()[0]
					table_input = [[var, c] for var, c in zip(input_variables, coefficient_list)]
					print(tabulate(table_input, headers=['Variable', 'Optimal Coefficient']))
				y_pred = model.predict(X_test)
				all_MSE.append(mean_squared_error(y_test, y_pred))
				print('\nMean squared error for '+biotype+': %.2f\n' % all_MSE[-1])

			print('-'*15 + 'Summary' + '-'*15)
			for biotype, mse in zip(args.biotypes.split(), all_MSE):
				print('{} MSE: {}'.format(biotype, mse))
		elif args.task == 'emotion':
			# file 1 contains fMRIs, file 2 contains subnum->login mapping, file 3 contains login->emotion recognition scores
			dataParser = DataParser(input_files=['./data/Neuroimaging/Reduced_biotype_imaging_data_s1.csv',\
												'./data/Webneuro/subNum_login_mapping.csv',\
												'./data/Webneuro/WebNeuro_Data_2018-10-21_21-54-37.csv' ], joinOn=['subNum', 'login'])
			X_train, X_test, y_train, y_test, input_variables = dataParser.get_data_splits(train_vars=['Act', 'PPI'], \
															label_var='getscp', print_input_vars=True)
			model = linear_model.LinearRegression()
			model.fit(X_train, y_train)
			if (args.show_coeffs):
				coefficient_list = model.coef_.tolist()[0]
				table_input = [[var, c] for var, c in zip(input_variables, coefficient_list)]
				print(tabulate(table_input, headers=['Variable', 'Optimal Coefficient']))
			y_pred = model.predict(X_test)
			print('\nMean squared error: %.2f\n' % mean_squared_error(y_test, y_pred))
		else:
			print("Provide a task for linear regression. See options in main.py")

	elif args.model == 'kmeans':
		print('Running K-Means...')
		dataParser = DataParser(input_files=['./data/Neuroimaging/Reduced_biotype_imaging_data_s1.csv'])
		X, input_variables = dataParser.get_input_vars(input_vars=['Act'])

		# this line does separate KMeans clustering than what sklearn does below this
		# this is just an example of how to use the hype graphing lib
		# graph = hype.plot(X, '.', n_clusters=8, legend=True, show=False) # plot clusters

		# KMeans using sklearn
		model = KMeans(n_clusters=args.n_clusters).fit(X)
		centers_list = model.cluster_centers_.tolist()
		center_values = [ [c[i] for c in centers_list] for i in range(len(input_variables))]
		table_input = [ [var[:4]+'...'+var[-6:]] + center_values[i] for i, var in enumerate(input_variables)]
		headers = ['Variable'] + ['Cluster {}'.format(x+1) for x in range(args.n_clusters)]
		print(tabulate(table_input, headers=headers))
	else:
		print('Plz provide a model arg')
