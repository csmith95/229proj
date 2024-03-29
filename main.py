import matplotlib.pyplot as plt
from data_parser import *
from sklearn import datasets, linear_model, svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, binarize
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.dummy import DummyClassifier
import sklearn
import argparse
from sklearn.cluster import KMeans
from tabulate import tabulate
import hypertools as hype
import pandas as pd
from sklearn.neural_network import MLPRegressor
from scipy.stats import mode
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import pdb
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


ALL_BIOTYPES = ['rumination', 'anxious_avoid', 'negative_bias', 'con_threat_dysreg', \
                'noncon_threat_dysreg', 'anhedonia', 'cog_dyscontrol', 'inattention']

# execution starts here
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Train a model.')
    # parser.add_argument('--model', type=str, help='Model to train & evaluate: \
    #                     linreg, rf, svm, neural', required=True)
    # parser.add_argument('--features', type=str, help='Features to include when training.  \
    #                     Available features: biotype, webneuro, medication, all', default = 'all')
    # parser.add_argument('--param_search', type=bool, help='Use GridSearchCV to test range of hyperparameters.  \
    #                     May take a good chunk of time.  Options are y/n', default = False)
    # parser.add_argument('--plot', type=bool)
    # parser.add_argument('--n_clusters', type=int, help='number of clusters for k-means')
    # # options for prediction task:
    #     # Linear Regression Tasks
    #         # (i) 'biotype' --> fits model to predict biotype scores from fMRI
    #         # (ii) 'emotion' --> fits model to predict average emotional recognition accuracy using fMRI
    #             # in progess (Conner)
    #         # (iii) depression -> fits model to predict score on depression indicators scl
    #             # TODO (haven't started)
    #     # Clustering Tasks
    #         # None (just breaks data into n_clusters)
    #         # TODO: interpret these results?
    #     # Neural Net Tasks
    #         # TODO
    #         # We can try to re-do the linear regression tasks using a neural net.
    #         # Also, This will probably be our best shot for analyzing the data over time.
    #         # We can use an LSTM to capture some patterns.
    # parser.add_argument('--task', type=str, help='prediction task')
    # parser.add_argument('--biotypes', type=str, help='reduced biotypes to consider as labels \
    #                                                 when fitting linear regression',
    #                                             nargs='+', default=' '.join(ALL_BIOTYPES))
    # parser.add_argument('--show_coeffs', type=bool, help='show coefficients for linear reg', \
    #                                                         default=False)
    # parser.add_argument('--verbose', type=bool, help='Display extra warning/info', default=False)
    # args = parser.parse_args()

    # dataParser = DataParser(input_files=['./data/Neuroimaging/Reduced_biotype_imaging_data_s1.csv'])
    # X, X_test, y_train, y_test, input_variables = dataParser.get_data_splits(train_vars=['Act', 'PPI'], \
         




# **************** this lost code is for LDA/PCA and is looking for a home in this project ***********

    # label_var='rumination_type_score', train_split=0.9)
    # dataParser = DataParser(input_files=['./data/Neuroimaging/Reduced_biotype_imaging_data_s1.csv'])
    # df = dataParser.data_frame
    # df = df.dropna()
    # depression_score_df = pd.read_csv('./data/depression_scores.csv')
    # merged = df.merge(depression_score_df, 'inner', on='subNum')
    # # y = pd.read_csv('./data/depression_scores.csv').values[:,1]
    # d_scores = merged['depression_score'].values
    # X = merged.drop(['depression_score', 'subNum'], axis=1).values
    # print(d_scores)
    # raise ValueError
    # scores = np.percentile(d_scores, [33, 66])
    # print(scores)
    # y = np.ones(d_scores.shape)
    # for i in range(d_scores.shape[0]):   
    #     if d_scores[i] <= scores[0]:
    #         y[i] = 0
    #     if d_scores[i] >= scores[1]:
    #         y[i] = 2 

    # pca = sklearn.decomposition.PCA(n_components=2)
    # X = pca.fit(X).transform(X)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # lda = LinearDiscriminantAnalysis(n_components=2)
    # X_train = lda.fit(X_train, y_train).transform(X_train)
    # plt.figure()
    # lw = 2
    # target_names = ['Bottom Tercile', 'Middle Tercile', 'Top Tercile']
    # colors = ['navy', 'turquoise', 'darkorange']
    # for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        # if i == 0:
        #     x = X[y <= scores[0], 0]
        #     y_r = X[y <= scores[0], 1]
        # if i == 1:
        #     indices = [a and b for a, b in zip(y >= scores[0], y <= scores[1])]
        #     x = X[indices, 0]
        #     y_r = X[indices, 1]
        # if i == 2:
        #     x = X[y >= scores[1], 0]
        #     y_r = X[y >= scores[1], 1]
        # plt.scatter(X[y==i,0], X[y==i,1], color=color, alpha=.8, lw=lw, label=target_name)

    # plt.scatter(X[:,0], X[:,1])
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.title('LDA of fMRI Data')
    # plt.show()
    # print(X_test.shape)
    # Z = lda.transform(X_test) #using the model to project Z
    # print()
    # print(Z)
    # z_labels = lda.predict(X_test) #gives you the predicted label for each sample
    # print(z_labels)
    # print('accuracy: ', np.sum(z_labels == y_test) / float(z_labels.shape[0]))



    dataParser = DataParser(input_files=['./data/Neuroimaging/Reduced_biotype_imaging_data_s1.csv'], verbose = args.verbose)
    df = dataParser.data_frame
    df = df.dropna()
    depression_score_df = pd.read_csv('./data/depression_scores.csv')
    if args.features == 'biotype':
        #Biotype
        merged = df.merge(depression_score_df, 'inner', on='subNum')
        mean_score = merged['depression_score'].mean()
        y_labels = [1 if x > mean_score else 0 for x in merged['depression_score']]
        y = merged['depression_score']
        #X = merged.loc[:, [b + '_type_score' for b in ALL_BIOTYPES]]
        X = merged.loc[:, [c for c in merged.columns.values if c[:2] in ['Ac', 'PP', 'RS']]]
    elif args.features == 'webneuro':
        # Webmneuro
        wanted_vars = list(pd.read_excel('./data/Webneuro/WebNeuro Data Dictionary.xlsx').VariableLabel)
        webneuro = pd.read_csv('./data/Webneuro/WebNeuro_Data_2018-10-21_21-54-37.csv')
        mapping = pd.read_csv('./data/Webneuro/subNum_login_mapping.csv')
        mapping = mapping.dropna()
        webneuro = webneuro.merge(mapping, 'inner', on='login').merge(depression_score_df, 'inner', on='subNum').dropna()
        webneuro['Gender'] = (webneuro['Gender'] == 'FEMALE').astype('int')
        webneuro = webneuro.groupby(['subNum']).mean().drop('suffix', axis=1)
        X = webneuro.drop('depression_score', axis=1)
        y = webneuro['depression_score']
    if args.features == 'medication':
        medication = pd.read_csv('./data/Medication/Medication_Data_share.csv').drop('24_month_arm_1', axis=1).dropna()
        #filter out medications ended before start of ENGAGE
        medication = medication[medication.END_DATE_Ndays_ENGAGE <= 0][['subNum', 'description']]
        medication['description'] = medication['description'].apply(lambda s: s.split(' ')[0])
        med_dummies = pd.get_dummies(medication, columns=['description']).groupby(['subNum'], as_index=False).sum()
        med_dummies = med_dummies.merge(depression_score_df, 'inner', on='subNum').drop('subNum', axis=1)
        y = med_dummies['depression_score']
        X = med_dummies.drop('depression_score', axis=1).clip(upper=1)
        print('Successfully obtained medication information.')
    if args.features == 'all':
        wanted_vars = list(pd.read_excel('./data/Webneuro/WebNeuro Data Dictionary.xlsx').VariableLabel)
        webneuro = pd.read_csv('./data/Webneuro/WebNeuro_Data_2018-10-21_21-54-37.csv')
        mapping = pd.read_csv('./data/Webneuro/subNum_login_mapping.csv')
        mapping = mapping.dropna()
        webneuro = webneuro.merge(mapping, 'inner', on='login').merge(depression_score_df, 'inner',
                                                        on='subNum').dropna().drop('suffix', axis=1)
        webneuro['Gender'] = (webneuro['Gender'] == 'FEMALE').astype('int')
        webneuro = webneuro.groupby(['subNum'], as_index=False).mean()

        medication = pd.read_csv('./data/Medication/Medication_Data_share.csv').drop('24_month_arm_1',
                                                                                     axis=1).dropna()
        # filter out medications ended before start of ENGAGE
        medication = medication[medication.END_DATE_Ndays_ENGAGE <= 0][['subNum', 'description']]
        medication['description'] = medication['description'].apply(lambda s: s.split(' ')[0])
        med_dummies = pd.get_dummies(medication, columns=['description']).groupby(['subNum'], as_index=False).sum()
        all = med_dummies.merge(webneuro, 'inner', on='subNum')
        X = all.drop(['subNum', 'depression_score'], axis=1)
        y = all['depression_score']

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


    elif args.model == 'svm':
        X = MinMaxScaler().fit_transform(X)
        y = np.asarray((y - y.min()) / (y.max() - y.min())).astype('float')
        X = SelectKBest(mutual_info_regression, k=17).fit_transform(X, y)

        #C = 0.7 if features == 'webneuro' else 0.7
        mse_model, mse_baseline = [], []
        tuned_params = []

        num_trials = 1
        for i in range(num_trials):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
            
            if args.param_search:
                tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], \
                                    'C': [.1,.2]}] #.5,1,2,5,10,50,100,200,1000
                                    #{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                clf = GridSearchCV(svm.SVR(), tuned_parameters, cv=5, n_jobs = -1) #use all processors
                
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
                mean_y_train = y_train.mean()
                mse_model.append(mean_squared_error(y_test, pred))
                mse_baseline.append(mean_squared_error(y_test, [mean_y_train] * len(y_test)))
                #pdb.set_trace()

                ax = plt.subplot(111)
                a = np.array(([1,2],[3,4]))
                im = ax.imshow(np.flip(a,axis=0))
                # create an axes on the right side of ax. The width of cax will be 5%
                # of ax and the padding between cax and ax will be fixed at 0.05 inch.
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                plt.ylim((3,7))
                plt.xlim((1,3))
                plt.show()
            else:
                clf = svm.SVR(kernel = 'rbf', C=1.0, gamma = 'auto')
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
                mean_y_train = y_train.mean()
                mse_model.append(mean_squared_error(y_test, pred))
                mse_baseline.append(mean_squared_error(y_test, [mean_y_train] * len(y_test)))

        print('Average mse for model:', sum(mse_model) / num_trials)
        print('Average mse for baseline:', sum(mse_baseline) / num_trials)
        print('Percent improvement:', 100 * (1.0 - sum(mse_model) / sum(mse_baseline)))


    elif args.model == 'rf':
        #show selected features
        print([f for f, i in zip(X.columns.values, SelectKBest(f_regression, k=3).fit(X,y).get_support()) if i])

        #normalize
        X = MinMaxScaler().fit_transform(X)
        y = np.asarray((y - y.min()) / (y.max() - y.min())).astype('float')

        # select features
        X = SelectKBest(f_regression, k=3).fit_transform(X, y)

        print('X shape:', X.shape)

        classify = False
        if classify:
            m = y.mean()
            y = [1.0 if elem > m else 0.0 for elem in y]

        mse_model, mse_baseline = [], []
        model_acc, baseline_acc = [], []
        num_trials = 1
        for i in range(num_trials):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i+1)
            if not classify:
                
                rf = RandomForestRegressor(random_state=1, n_estimators=1000, max_features=2, max_depth=2)
                rf.fit(X_train, y_train)
                lr = linear_model.LinearRegression()
                lr.fit(X_train, y_train)
                nn = MLPRegressor(hidden_layer_sizes=(10, 10, 10), solver='lbfgs', alpha=1e-6, max_iter=10000000)
                nn.fit(X_train, y_train)
                pred = rf.predict(X_test)

                mean_y_train = y_train.mean()
                mse_model.append(mean_squared_error(y_test, pred))
                mse_baseline.append(mean_squared_error(y_test, [mean_y_train] * len(y_test)))
            else:
                rf_clf = RandomForestClassifier(random_state=1, n_estimators=1000, max_features=2, max_depth=2)
                rf_clf.fit(X_train, y_train)

                pred = rf_clf.predict(X_test)
                model_acc.append(accuracy_score(y_test, pred))
                baseline_acc.append(accuracy_score(y_test, [mode(y_train)[0] for elem in range(len(y_test))]))


            # print(sum(y_train) / len(y_train))
            # print(sum(y_test) / len(y_test))

            #print()

            # model = linear_model.LinearRegression()
            # model.fit(X, y_train)
        if not classify:
            print('Average mse for model:', sum(mse_model) / num_trials)
            print('Average mse for baseline:', sum(mse_baseline) / num_trials)
            print('Percent improvement:', 100 * (1.0 - sum(mse_model) / sum(mse_baseline)))
        else:
            print('Classifcation accuracy for model:', np.mean(model_acc))
            print('Classifcation accuracy for baseline:', np.mean(baseline_acc))
            rf = RandomForestClassifier(random_state=1, n_estimators=1000, max_features=2, max_depth=2)
            dummy = DummyClassifier(strategy='most_frequent', random_state=0)
            avg_cross_val_score = np.mean(cross_val_score(rf, X, y, cv=5))
            avg_baseline_cross_val_score = np.mean(cross_val_score(dummy, X, y, cv=5))
            print('Average cross validation score:', avg_cross_val_score)
            print('Average baseline cross validation score:', avg_baseline_cross_val_score)
            print('Percent Improvement:', 100 * (avg_cross_val_score / avg_baseline_cross_val_score - 1))

    else:
        print('Plz provide a model arg')
