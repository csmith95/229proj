import matplotlib.pyplot as plt
from data_parser import *
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import argparse
from sklearn.cluster import KMeans
from tabulate import tabulate
import hypertools as hype


# execution starts here
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train a model.')
	parser.add_argument('--model', type=str, help='type of model to train/evaluate', required=True)
	parser.add_argument('--n_clusters', type=int, help='number of clusters for k-means')
	args = parser.parse_args()

	dataParser = DataParser(input_files=['./data/Neuroimaging/Reduced_biotype_imaging_data_s1.csv'])

	if args.model == 'linreg':
		print('\nTraining a Linear Regression Model...')
		X_train, X_test, y_train, y_test, input_variables = dataParser.get_data_splits(train_vars=['Act'], label_var='rumination_type_score')
		model = linear_model.LinearRegression()
		model.fit(X_train, y_train)
		coefficient_list = model.coef_.tolist()[0]
		table_input = [[var, c] for var, c in zip(input_variables, coefficient_list)]
		y_pred = model.predict(X_test)
		print(tabulate(table_input, headers=['Variable', 'Optimal Coefficient']))
		print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
	elif args.model == 'kmeans':
		X, input_variables = dataParser.get_input_vars(input_vars=['Act'])
		# this line does separate KMeans clustering than what sklearn does below this
		# this is just an example of how to use the hype graphing lib
		graph = hype.plot(X, '.', n_clusters=8, legend=True, show=False) # plot clusters

		# KMeans using sklearn
		model = KMeans(n_clusters=args.n_clusters).fit(X)
		centers_list = model.cluster_centers_.tolist()
		center_values = [ [c[i] for c in centers_list] for i in range(len(input_variables))]
		table_input = [ [var[:4]+'...'+var[-6:]] + center_values[i] for i, var in enumerate(input_variables)]
		headers = ['Variable'] + ['Cluster {}'.format(x+1) for x in range(args.n_clusters)]
		print(tabulate(table_input, headers=headers))
	else:
		print('Plz provide a model arg')
