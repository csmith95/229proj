import matplotlib.pyplot as plt
from data_parser import *
from sklearn import datasets, linear_model, svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import argparse
from sklearn.cluster import KMeans
from tabulate import tabulate
import hypertools as hype
import pandas as pd
from sklearn.neural_network import MLPRegressor

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
                # in progess (Conner)
            # (iii) depression -> fits model to predict score on depression indicators scl
                # TODO (haven't started)
        # Clustering Tasks
            # None (just breaks data into n_clusters)
            # TODO: interpret these results?
        # Neural Net Tasks
            # TODO
            # We can try to re-do the linear regression tasks using a neural net.
            # Also, This will probably be our best shot for analyzing the data over time.
            # We can use an LSTM to capture some patterns.
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
    elif args.model == 'neural':
        model = MLPRegressor(hidden_layer_sizes=(15, 5, 5), alpha=1e-5, max_iter=5000)
        print('Training a Neural Network Model...')
        if args.task == 'biotype':
            dataParser = DataParser(input_files=['./data/Neuroimaging/Reduced_biotype_imaging_data_s1.csv'])
            all_MSE = []
            for i, biotype in enumerate(args.biotypes.split()):
                biotype_variable_name = '{}_type_score'.format(biotype)
                X_train, X_test, y_train, y_test, input_variables = dataParser.get_data_splits(
                    train_vars=['Act', 'PPI', 'RS_'], \
                    label_var=biotype_variable_name, print_input_vars=(i == 0))
                model.fit(X_train, y_train)
                if args.show_coeffs:
                    coefficient_list = model.coef_.tolist()[0]
                    table_input = [[var, c] for var, c in zip(input_variables, coefficient_list)]
                    print(tabulate(table_input, headers=['Variable', 'Optimal Coefficient']))
                y_pred = model.predict(X_test)
                all_MSE.append(mean_squared_error(y_test, y_pred))
                print('\nMean squared error for ' + biotype + ': %.2f' % all_MSE[-1])

                # Error analysis: see what we would get by predicting the mean accuracy
                mean_acc = np.mean(y_train)
                mean_pred = mean_acc * np.ones_like(y_test)
                print('"Guessing" mean squared error: %.2f\n' % mean_squared_error(y_test, mean_pred))

            print('-' * 15 + 'Summary' + '-' * 15)
            for biotype, mse in zip(args.biotypes.split(), all_MSE):
                print('{} MSE: {}'.format(biotype, mse))
        elif args.task == 'emotion':
            # file 1 contains fMRIs, file 2 contains subnum->login mapping, file 3 contains login->emotion recognition scores
            dataParser = DataParser(input_files=['./data/Neuroimaging/Reduced_biotype_imaging_data_s1.csv', \
                                                 './data/Webneuro/subNum_login_mapping.csv', \
                                                 './data/Webneuro/WebNeuro_Data_2018-10-21_21-54-37.csv'],
                                    joinOn=['subNum', 'login'])
            dataParser.filter_data(key='suffix', value=1)
            X_train, X_test, y_train, y_test, input_variables = dataParser.get_data_splits(train_vars=['Act'], \
                                                                                           label_var='getscp',
                                                                                           print_input_vars=True)
            model.fit(X_train, y_train.reshape((-1,)))
            if (args.show_coeffs):
                coefficient_list = model.coef_.tolist()[0]
                table_input = [[var, c] for var, c in zip(input_variables, coefficient_list)]
                print(tabulate(table_input, headers=['Variable', 'Optimal Coefficient']))
            y_pred = model.predict(X_test)
            print('\nModel Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))

            # Error analysis: see what we would get by predicting the mean accuracy
            mean_acc = np.mean(y_train)
            mean_pred = mean_acc * np.ones_like(y_test)
            print('"Guessing" mean squared error: %.2f\n' % mean_squared_error(y_test, mean_pred))

        else:
            print("Provide a task for neural net. See options in main.py")
    elif args.model == 'rf':
        dataParser = DataParser(input_files=['./data/Neuroimaging/Reduced_biotype_imaging_data_s1.csv'])
        df = dataParser.data_frame
        df = df.dropna()
        depression_score_df = pd.read_csv('./data/depression_scores.csv')
        features = 'biotype'

        if features == 'biotype':
            #Biotype
            merged = df.merge(depression_score_df, 'inner', on='subNum')
            mean_score = merged['depression_score'].mean()
            y_labels = [1 if x > mean_score else 0 for x in merged['depression_score']]
            y = merged['depression_score']
            #X = merged.loc[:, [b + '_type_score' for b in ALL_BIOTYPES]]
            X = merged.loc[:, [c for c in merged.columns.values if c[:2] in ['Ac', 'PP', 'RS']]]
        elif features == 'webneuro':
            # Webmneuro
            wanted_vars = list(pd.read_excel('./data/Webneuro/WebNeuro Data Dictionary.xlsx').VariableLabel)
            webneuro = pd.read_csv('./data/Webneuro/WebNeuro_Data_2018-10-21_21-54-37.csv')
            mapping = pd.read_csv('./data/Webneuro/subNum_login_mapping.csv')
            mapping = mapping.dropna()
            webneuro = webneuro.merge(mapping, 'inner', on='login').merge(depression_score_df, 'inner', on='subNum').dropna()
            webneuro['Gender'] = (webneuro['Gender'] == 'FEMALE').astype('int')
            webneuro = webneuro.select_dtypes(['number']).drop('suffix', axis=1)
            X = webneuro.drop('depression_score', axis=1)
            y = webneuro['depression_score']
        if features == 'medication':
            medication = pd.read_csv('./data/Medication/Medication_Data_share.csv').drop('24_month_arm_1', axis=1).dropna()
            #filter out medications ended before start of ENGAGE
            medication = medication[medication.END_DATE_Ndays_ENGAGE <= 0][['subNum', 'description']]
            medication['description'] = medication['description'].apply(lambda s: s.split(' ')[0])
            med_dummies = pd.get_dummies(medication, columns=['description']).groupby(['subNum'], as_index=False).sum()
            med_dummies = med_dummies.merge(depression_score_df, 'inner', on='subNum').drop('subNum', axis=1)
            y = med_dummies['depression_score']
            X = med_dummies.drop('depression_score', axis=1).clip(upper=1)
            print('Successfully obtained medication information.')



        #normalize
        X = StandardScaler().fit_transform(X)

        mse_model, mse_baseline = [], []
        num_trials = 10
        for i in range(num_trials):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

            C = 0.7 if features == 'webneuro' else 0.2

            rf = RandomForestRegressor(random_state=1, n_estimators=1000, max_features=2, max_depth=2)
            rf.fit(X_train, y_train)
            lr = linear_model.LinearRegression()
            lr.fit(X_train, y_train)
            clf = svm.SVR(C=C)
            clf.fit(X_train, y_train)

            pred = clf.predict(X_test)
            mean_y_train = y_train.mean()

            # print(sum(y_train) / len(y_train))
            # print(sum(y_test) / len(y_test))

            #print(accuracy_score(y_test, pred))
            mse_model.append(mean_squared_error(y_test, pred))
            mse_baseline.append(mean_squared_error(y_test, [mean_y_train] * len(y_test)))
            # model = linear_model.LinearRegression()
            # model.fit(X, y_train)

        print('Average mse for model:', sum(mse_model) / num_trials)
        print('Average mse for baseline:', sum(mse_baseline) / num_trials)


    else:
        print('Plz provide a model arg')
