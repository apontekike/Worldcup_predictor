from bottle import route, run , template, request
from test2 import *
from world_cup_predictor import *

@route('/')
def index():
    return template('index', winner=-1)

@route('/predict')
def predict():
    A_name = request.query.team_A
    B_name = request.query.team_B
 

    winner = int(clf.predict(gen_X(teams_stats[A_name],teams_stats[B_name]))[0])
    print("*************************")
    print(winner)
    return template('index', winner= int(winner), tA = A_name,  tB = B_name)




run(host="localhost", port=8080, debug=True, reloader=True)

teams_str="Qatar Equador Senegal Netherlands England Iran USA Wales Argentina Saudi_Arabia Mexico Poland France Australia Denmark Tunisia Spain Costa_Rica Germany Japan Belgium Canada Morroco Croatia Brazil Serbia Switzerland Cameroon Portugal Ghana Uruguay Korea_Republic"
teams_arr=['Qatar', 'Equador', 'Senegal', 'Netherlands', 'England', 'Iran', 'USA', 'Wales', 'Argentina', 'Saudi_Arabia', 'Mexico', 'Poland', 'France', 'Australia', 'Denmark', 'Tunisia', 'Spain', 'Costa_Rica', 'Germany', 'Japan', 'Belgium', 'Canada', 'Morroco', 'Croatia', 'Brazil', 'Serbia', 'Switzerland', 'Cameroon', 'Portugal', 'Ghana', 'Uruguay', 'Korea_Republic']


