from __future__ import division

import box_plots
import player_comparison
import correct_predictions
import aggregate_comparisons 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import PassiveAggressiveClassifier
import re
import numpy as np
import os
from matplotlib import pyplot as plt
#import seaborn as sns



# finding optimal parameters for Stochastic Gradient Descent - Logistic Regression with Elastic Net
def stoch_grad_hit(to_predict_hitters,x_hitters, hitter_predictions):
	best_params = []

	for col in to_predict_hitters:
	    y = hitter_predictions[col].tolist()
	    x_train, x_test, y_train, y_test = train_test_split(x_hitters, y)
	    
	    svr = linear_model.SGDClassifier()
	    
	    parameters = {'loss':['log'], 'penalty': ['elasticnet'],'alpha': [.0001, .001, .01, .1], \
	                  'l1_ratio': [0, 0.15, 0.25, 1., .5], 'fit_intercept': [True]}
	    clf = GridSearchCV(svr, parameters)
	    clf.fit(x_train, y_train)
	    best_params.append({col:clf.best_params_})
	    
	return best_params

def stoch_grad_pitch(to_predict_pitchers,x_pitchers, pitcher_predictions):
	best_params = []

	for col in to_predict_pitchers:
	    y = pitcher_predictions[col].tolist()
	    x_train, x_test, y_train, y_test = train_test_split(x_pitchers, y)
	    
	    svr = linear_model.SGDClassifier()
	    
	    parameters = {'loss':['log'], 'penalty': ['elasticnet'],'alpha': [.0001, .001, .01, .1], \
	                   'l1_ratio': [0, 0.15, 0.25, 1., .5], 'fit_intercept': [True]}
	    clf = GridSearchCV(svr, parameters)
	    clf.fit(x_train, y_train)
	    best_params.append({col:clf.best_params_})
	    
	return best_params

# finding optimal parameters for passive aggressive algorithm
def pa_hit(to_predict_hitters,x_hitters, hitter_predictions):
	best_params = []

	for col in to_predict_hitters:
	    y = hitter_predictions[col].tolist()
	    x_train, x_test, y_train, y_test = train_test_split(x_hitters, y)
	    
	    passAggress = PassiveAggressiveClassifier()
	    
	    parameters = {'C':[.0001, .001, .01, .1, 1.0], 'fit_intercept':[True], 'n_iter':[5, 10], 'shuffle':[True],\
	                  'loss':['hinge'], 'n_jobs':[1], 'random_state':[None], 'class_weight':[None]}
	    clf = GridSearchCV(passAggress, parameters)
	    clf.fit(x_train, y_train)
	    best_params.append({col:clf.best_params_})
	    
	return best_params

def pa_pitch(to_predict_pitchers,x_pitchers, pitcher_predictions):
	best_params = []

	for col in to_predict_pitchers:
	    y = pitcher_predictions[col].tolist()
	    x_train, x_test, y_train, y_test = train_test_split(x_pitchers, y)
	    
	    passAggress = PassiveAggressiveClassifier()
	    
	    parameters = {'C':[.0001, .001, .01, .1, 1.0], 'fit_intercept':[True], 'n_iter':[5, 10], 'shuffle':[True],\
	                  'loss':['hinge'], 'n_jobs':[1], 'random_state':[None], 'class_weight':[None]}
	    clf = GridSearchCV(passAggress, parameters)
	    clf.fit(x_train, y_train)
	    best_params.append({col:clf.best_params_})
	    
	return best_params
