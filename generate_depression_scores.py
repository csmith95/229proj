import pandas as pd

df = pd.read_csv('./data/PAMFRI/basline_2_5_all copy.csv')
cols = ['bv_scl' + str(i) for i in range(1, 21)]
depression_scores = df.loc[:, cols]
depression_scores['depression_score'] = depression_scores.mean(axis=1)
depression_scores['subNum'] = df['StudyID']
depression_scores[['subNum', 'depression_score']].to_csv('./data/depression_scores.csv', index=False)