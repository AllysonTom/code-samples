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

def accuracy_lists_hitters(method, best_params, x_hitters, hitter_predictions, to_predict_hitters):
    # create a set of lists that will store accuracies

    accuracy_list = []
    accuracy_list_AVG = []
    accuracy_list_HR = []
    accuracy_list_R = []
    accuracy_list_RBI = []
    accuracy_list_SB = []

    for i in xrange(10):
        j=0
        for col in to_predict_hitters:
            y = hitter_predictions[col].tolist()
            x_train, x_test, y_train, y_test = train_test_split(x_hitters, y)

            chosen_method = method(**best_params[j][col])
            chosen_method.fit(x_train, y_train)
            acc = metrics.accuracy_score(y_test, chosen_method.predict(x_test))
            accuracy_list.append(acc)

            if col == 'correct_AVG':
                accuracy_list_AVG.append(acc)
            elif col == 'correct_HR':
                accuracy_list_HR.append(acc)
            elif col == 'correct_R':
                accuracy_list_R.append(acc)
            elif col == 'correct_RBI':
                accuracy_list_RBI.append(acc)
            elif col == 'correct_SB':
                accuracy_list_SB.append(acc)
            j+=1
            
    print "%-15s" % 'overall average', np.mean(accuracy_list)
    print "%-15s" % 'correct_AVG', np.mean(accuracy_list_AVG)
    print "%-15s" % 'correct_HR', np.mean(accuracy_list_HR)
    print "%-15s" % 'correct_R', np.mean(accuracy_list_R)
    print "%-15s" % 'correct_RBI', np.mean(accuracy_list_RBI)
    print "%-15s" % 'correct_SB', np.mean(accuracy_list_SB)

def accuracy_lists_pitchers(method, best_params, x_pitchers, pitcher_predictions, to_predict_pitchers):
    # implementing using optimal parameters for our pitchers

    accuracy_list_pitchers = []
    accuracy_list_pitchers_ERA = []
    accuracy_list_pitchers_K = []
    accuracy_list_pitchers_W = []
    accuracy_list_pitchers_WHIP = []

    for i in xrange(10):
        j=0
        for col in to_predict_pitchers:
            y = pitcher_predictions[col].tolist()
            x_train, x_test, y_train, y_test = train_test_split(x_pitchers, y)

            chosen_method = method(**best_params[j][col])
            chosen_method.fit(x_train, y_train)
            acc = metrics.accuracy_score(chosen_method.predict(x_test), y_test)
            accuracy_list_pitchers.append(acc)

            if col == 'correct_ERA':
                accuracy_list_pitchers_ERA.append(acc)
            elif col == 'correct_K':
                accuracy_list_pitchers_K.append(acc)
            elif col == 'correct_W':
                accuracy_list_pitchers_W.append(acc)
            elif col == 'correct_WHIP':
                accuracy_list_pitchers_WHIP.append(acc)
            j+=1  

    print "%-15s" % 'overall average', np.mean(accuracy_list_pitchers)
    print "%-15s" % 'correct_ERA', np.mean(accuracy_list_pitchers_ERA)
    print "%-15s" % 'correct_K', np.mean(accuracy_list_pitchers_K)
    print "%-15s" % 'correct_W', np.mean(accuracy_list_pitchers_W)
    print "%-15s" % 'correct_WHIP', np.mean(accuracy_list_pitchers_WHIP)
