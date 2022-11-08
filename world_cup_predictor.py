import numpy as np 
import pandas as pd

df = pd.read_csv("international_matches.csv")

# Cleaning training data Set
df.sort_values('date', ascending=False)
df = df.drop(['home_team_total_fifa_points','away_team_total_fifa_points'
                ,'neutral_location','home_team_score','away_team_score',
                'shoot_out','tournament','city','date'
                ,'home_team_continent','away_team_continent'], axis=1)

df.dropna(inplace=True)

df = df[df.home_team_result != 'Draw']

df.loc[df['home_team_result'] == 'Win', 'A_result']  = 1
df.loc[df['home_team_result'] == 'Lose', 'A_result']  = 0


df = df.drop(['home_team_result'], axis=1)

df = df.drop(['country','home_team','away_team'], axis=1)

df.rename(columns={  'home_team_fifa_rank':'fifa_ranking_A',
        'away_team_fifa_rank' : 'fifa_ranking_B', 'home_team_goalkeeper_score' : 'goal_keeper_score_A', 
        'away_team_goalkeeper_score':'goal_keeper_score_B', 'home_team_mean_defense_score' : 'mean_defense_score_A',
        'away_team_mean_defense_score' : 'mean_defense_score_B', 'home_team_mean_midfield_score' : 'mean_midfield_score_A',
        'away_team_mean_midfield_score' : 'mean_midfield_score_B', 'home_team_mean_offense_score':'mean _offense_score_A', 
        'away_team_mean_offense_score' : 'mean _offense_score_B'},inplace=True)

df.drop(["fifa_ranking_B", "fifa_ranking_A"],axis=1)

df.reset_index(drop=True, inplace=True)


#Spliting the data set into Train, Validation and Test set
data = len(df.index)
Train_set = df.iloc[:int(data*0.90)]
val_set = df.iloc[int(data*0.90):]

val_set.reset_index(drop=True,inplace=True)

data = len(val_set.index)
Val_set = val_set.iloc[:int(data*0.50)]
Test_set = val_set.iloc[int(data*0.50):]

Val_set.reset_index(drop=True,inplace=True)
Test_set.reset_index(drop=True,inplace=True)

# Saving the Data Set in a CSV
Train_set.to_csv('Train_Data_Set')
Val_set.to_csv('Validation_Data_Set')
Test_set.to_csv('Test_Data_Set')

#Getting Y_train 
Y_train = Train_set['A_result'].to_numpy()
Y_test = Test_set['A_result'].to_numpy()
Y_val = Val_set['A_result'].to_numpy()

#Getting X_Train, X_val and X_test
X_train = Train_set.drop(['A_result'],axis=1)
X_val = Val_set.drop(['A_result'],axis=1)
X_test = Test_set.drop(['A_result'],axis=1)

X_train = X_train.to_numpy()
X_val = X_val.to_numpy()
X_test = X_test.to_numpy()


import sklearn
from sklearn import svm
from sklearn import metrics



clf = svm.SVC(kernel="linear")
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
weights = clf.coef_

print(weights)

acc = metrics.accuracy_score(Y_test, Y_pred)

print(acc)
