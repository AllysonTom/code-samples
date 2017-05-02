from __future__ import division
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn import preprocessing
from xgboost import XGBClassifier as xgb
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
import box_plots
import player_comparison
import correct_predictions
import aggregate_comparisons 
import pandas as pd
import re
import numpy as np
import os
from matplotlib import pyplot as plt


def lr_hitters_params(to_predict_hitters, x_hitters, hitter_predictions):

	# finding optimal parameters for logistic regression

	best_params = []

	for col in to_predict_hitters:
	    y = hitter_predictions[col].tolist()
	    x_train, x_test, y_train, y_test = train_test_split(x_hitters, y)
	    
	    log_reg = linear_model.LogisticRegression()
	    
	    parameters = {'C':[0.5, 1.0, 1.5], 'class_weight':[None, 'balanced'], \
	                  'solver':['newton-cg', 'liblinear', 'lbfgs', 'sag'],\
	                  'max_iter':[100, 150]}
	    
	    clf = GridSearchCV(log_reg, parameters)
	    clf.fit(x_train, y_train)
	    best_params.append({col:clf.best_params_})

	return best_params
def lr_pitchers_params(to_predict_pitchers, x_pitchers, pitcher_predictions):
	# logistic regression pitchers
	# finding best logistic regression parameters for our pitchers

	best_params = []

	for col in to_predict_pitchers:
	    y = pitcher_predictions[col].tolist()
	    x_train, x_test, y_train, y_test = train_test_split(x_pitchers, y)
	    
	    log_reg = linear_model.LogisticRegression()
	    parameters = {'C':[0.5, 1.0, 1.5], 'class_weight':[None, 'balanced'], 'random_state':[None, 5], \
	                  'solver':['liblinear', 'newton-cg', 'lbfgs', 'sag']}
	    clf = GridSearchCV(log_reg, parameters)
	    clf.fit(x_train, y_train)
	    best_params.append({col:clf.best_params_})
	    
	return best_params


####### Random Forests Optimal parameters ###########
def rf_hitters_params(to_predict_hitters, x_hitters, hitter_predictions):
	### running random forests for hitters

	best_params = []

	for col in to_predict_hitters:
	    
	    # looping through each of the desired prediction columns for hitters and splitting into test/train
	    y = hitter_predictions[col].tolist()
	    x_train, x_test, y_train, y_test = train_test_split(x_hitters, y)
	    
	    # instantiate randomForest object and then pass in different potential arguments
	    randomForest = RandomForestClassifier()
	    parameters = {'n_estimators':[10, 15, 20], 'criterion':['gini', 'entropy'], \
	                  'min_samples_leaf':[1, 5], 'max_features':['auto', 'sqrt'],\
	                  'max_leaf_nodes':[None, 5], 'bootstrap':[True], 'n_jobs':[1, -1]}
	    
	    # run the GridSearch to find optimal arguments
	    clf = GridSearchCV(randomForest, parameters)
	    clf.fit(x_train, y_train)
	    best_params.append({col:clf.best_params_})
	    
	return best_params

def rf_pitchers_params(to_predict_pitchers, x_pitchers, pitcher_predictions):
	# running random forests for pitchers

	best_params = []

	for col in to_predict_pitchers:
	    y = pitcher_predictions[col].tolist()
	    x_train, x_test, y_train, y_test = train_test_split(x_pitchers, y)
	    
	    randomForest = RandomForestClassifier()
	    parameters = {'n_estimators':[10, 15, 20], 'criterion':['gini', 'entropy'], \
	                  'min_samples_leaf':[1, 5], 'max_features':['auto', 'sqrt'],\
	                  'max_leaf_nodes':[None, 5], 'bootstrap':[True], 'n_jobs':[1, -1]}
	    clf = GridSearchCV(randomForest, parameters)
	    clf.fit(x_train, y_train)
	    best_params.append({col:clf.best_params_})
	    
	return best_params


####### XGBoost Optimal parameters ###########

### running a grid search cross validation on XGBoost for hitters to obtain the best parameters

def xg_hitters_params(to_predict_hitters, x_hitters, hitter_predictions):
	best_params = []

	for col in to_predict_hitters:
	    
	    y = hitter_predictions[col].tolist()
	    x_train, x_test, y_train, y_test = train_test_split(x_hitters, y)
	    
	    xgb_classifier = xgb()
	    parameters = {'max_depth': [3,5,9], 'learning_rate': [.1,.4], "n_estimators": [250,350], \
	                'reg_lambda': [1,4]}
	    clf = GridSearchCV(xgb_classifier, parameters)
	    clf.fit(x_train, y_train)
	    best_params.append({col:clf.best_params_})
	    
	return best_params

def xg_pitchers_params(to_predict_pitchers, x_pitchers, pitcher_predictions):
	### running a grid search cross validation on XGBoost for pitchers to obtain the best parameters

	best_params = []

	for col in to_predict_pitchers:
	    y = pitcher_predictions[col].tolist()
	    x_train, x_test, y_train, y_test = train_test_split(x_pitchers, y)
	    
	    xgb_classifier = xgb()
	    parameters = {'max_depth': [3,5,9], 'learning_rate': [.05,.1], "n_estimators": [250,350], \
	                'reg_lambda': [1,3,6]}
	    clf = GridSearchCV(xgb_classifier, parameters)
	    clf.fit(x_train, y_train)
	    best_params.append({col:clf.best_params_})
	    
	return best_params


####### QDA Optimal parameters ###########

def qda_hitters_params(to_predict_hitters, x_hitters, hitter_predictions):
	# finding optimal parameters for QDA

	best_params = []

	for col in to_predict_hitters:
	    y = hitter_predictions[col].tolist()
	    x_train, x_test, y_train, y_test = train_test_split(x_hitters, y)
	    
	    qda = QuadraticDiscriminantAnalysis()
	    parameters = {'priors':[None], 'reg_param':[0.0], 'store_covariances':[False],\
	                  'tol':[0.0000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]}
	    clf = GridSearchCV(qda, parameters)
	    clf.fit(x_train, y_train)
	    best_params.append({col:clf.best_params_})
	    
	return best_params


def qda_pitchers_params(to_predict_pitchers, x_pitchers, pitcher_predictions):
	# finding best qda parameters for our pitchers

	best_params = []

	for col in to_predict_pitchers:
	    y = pitcher_predictions[col].tolist()
	    x_train, x_test, y_train, y_test = train_test_split(x_pitchers, y)
	    
	    qda = QuadraticDiscriminantAnalysis()
	    parameters = {'priors':[None], 'reg_param':[0.0], 'store_covariances':[False],\
	                  'tol':[0, 0.0000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]}
	    clf = GridSearchCV(qda, parameters)
	    clf.fit(x_train, y_train)
	    best_params.append({col:clf.best_params_})
	    
	return best_params


####### SVM Optimal parameters ###########

def svm_hitters_params(to_predict_hitters, x_hitters, hitter_predictions):
	# create lists of parameters to search through
	c = [10**i for i in np.arange(-3,3)]
	gamma = c
	poly_coeff0 = [10**i for i in np.arange(0,3)]

	# finding optimal parameters for svm

	best_params = []

	# preprocess the x values
	x_hitters = preprocessing.scale(x_hitters)

	for col in to_predict_hitters:
	    y = hitter_predictions[col].tolist()
	    x_train, x_test, y_train, y_test = train_test_split(x_hitters, y)
	    
	    svr = svm.SVC()
	    
	    parameters = {'kernel':['rbf'], 'gamma': gamma,'coef0': poly_coeff0}
	    clf = GridSearchCV(svr, parameters)
	    clf.fit(x_train, y_train)
	    best_params.append({col:clf.best_params_})
	    
	return best_params


def svm_pitchers_params(to_predict_pitchers, x_pitchers, pitcher_predictions):
	# svm pitchers
	# create lists of parameters to search through
	c = [10**i for i in np.arange(-3,3)]
	gamma = c
	poly_coeff0 = [10**i for i in np.arange(0,3)]
	# finding best svm parameters for our pitchers

	best_params = []

	for col in to_predict_pitchers:
	    y = pitcher_predictions[col].tolist()
	    x_train, x_test, y_train, y_test = train_test_split(x_pitchers, y)
	    
	    svr = svm.SVC()
	    
	    parameters = {'kernel':['rbf'], 'gamma': gamma,'coef0': poly_coeff0}
	    clf = GridSearchCV(svr, parameters)
	    clf.fit(x_train, y_train)
	    best_params.append({col:clf.best_params_})
	    
	return best_params
