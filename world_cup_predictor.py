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



#TEAM STATS
Qatar=[50 , 78, 77, 77, 77]
Equador=[44, 75, 75, 75, 74]
Senegal=[ 18, 79, 76, 77, 86]
Netherlands=[ 8, 82, 81, 83, 79]
England=[ 5, 86, 83, 83, 83]
Iran=[ 20, 81, 73, 72, 74]
USA=[ 16, 77, 78, 76, 79]
Wales=[ 19, 78, 76, 77, 76]
Argentina=[ 3, 86, 84, 82, 84]
Saudi_Arabia=[ 51, 71, 72, 71, 71]
Mexico=[13, 79, 77, 77, 84]
Poland=[ 26, 81, 75, 75, 86]
France=[ 4, 85, 83, 83, 87]
Australia=[38 , 72, 72, 71, 80]
Denmark=[ 10, 77, 83, 80, ]
Tunisia=[ 30, 72, 75, 71, 73]
Spain=[ 7, 83, 85, 83, 84]
Costa_Rica=[ 31, 73, 73, 74, 87]
Germany=[ 11, 84, 85, 82, 90]
Japan=[ 24, 75, 77, 76, 79]
Belgium=[ 2, 84, 81, 79, 90]
Canada=[ 41, 75, 78, 72, 75]
Morroco=[ 22, 79, 73, 78, 82]
Croatia=[ 12, 80, 83, 78, 81]
Brazil=[ 1, 85, 85, 83, 89]
Serbia=[ 21, 80, 80, 75, 81]
Switzerland=[ 15, 77, 78, 78, 86]
Cameroon=[ 43, 75, 75, 72, 70 ]
Portugal=[ 9, 84, 84, 84, 83]
Ghana=[ 61, 81, 76, 75, 76]
Uruguay=[ 14, 81, 82, 79, 78]
Korea_Republic=[ 28, 79, 74, 75, 78]


teams_stats = {"Qatar":Qatar,
"Equador":Equador,
"Senegal":Senegal,
"Netherlands":Netherlands,
"England":England,
"Iran":Iran,
"USA":USA,
"Wales":Wales,
"Argentina":Argentina,
"Saudi_Arabia":Saudi_Arabia,
"Mexico":Mexico,
"Poland":Poland,
"France":France,
"Australia":Australia,
"Denmark":Denmark,
"Tunisia":Tunisia,
"Spain":Spain,
"Costa_Rica":Costa_Rica,
"Germany":Germany,
"Japan":Japan,
"Belgium":Belgium,
"Canada":Canada,
"Morroco":Morroco,
"Croatia":Croatia,
"Brazil":Brazil,
"Serbia":Serbia,
"Switzerland":Switzerland,
"Cameroon":Cameroon,
"Portugal":Portugal,
"Ghana":Ghana,
"Uruguay":Uruguay,
"Korea_Republic":Korea_Republic}


import sklearn
from sklearn import svm
from sklearn import metrics



clf = svm.SVC(kernel="linear")
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
weights = clf.coef_

print(weights)

acc = metrics.accuracy_score(Y_test, Y_pred)
#,fifa_ranking_A,fifa_ranking_B,goal_keeper_score_A,goal_keeper_score_B,mean_defense_score_A,mean _offense_score_A,mean_midfield_score_A,mean_defense_score_B,mean _offense_score_B,mean_midfield_score_B,A_result

print(acc)
def gen_X(A,B):
        return [[A[0],B[0],A[4],B[4],A[3],A[1],A[2],B[3],B[1],B[2]]]

print(clf.predict(gen_X(France, Qatar))[0])
