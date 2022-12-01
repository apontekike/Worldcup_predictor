# ASL_alphabet_classifier

Final project for McGill AI Society Intro to ML Bootcamp (Fall 2022). 

Training data retrieved from [Kaggle](https://www.kaggle.com/datasets/brenda89/fifa-world-cup-2022).

## Project description

This world cup 2022 predictor, is given an input of two teams participating in the 2022 World Cup, and outputs which teams is going to win the game, based on the fifa23 game teams rating and fifa ranking. 

We built the classification model using Sklearn library and the web app's backend using Bottle. We retrieved training data from Kaggle.

## Running the app

To run the web app, install the packages pandas, Sklearn, and Bottle. Then run

```
python WebApp.py
```

Lastly, open a browser and navigate to your http://localhost:8000.

## Repository organization

This repository contains the scripts used to both train the model and build the web app.

1. deliverables/
	* deliverables submitted to the MAIS Intro to ML Bootcamp organizers
2. views/
	* Contains the html template of the web application.
	* The styling of the page is embedde in the same file at the top.
5. world_cup_predictor.py
	* python class to store model and its methods for web app
6. web_app.py
	* main python script to instantiate Bottle server
