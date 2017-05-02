from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier as xgb
import numpy as np
from operator import itemgetter
import collections
import pandas as pd

def statistic_percentage(xgb_hitters_params, xgb_pitchers_params, rforest_pitchers_params, hitter_predictions_2017, x_hitters, x_hitters2017,
pitcher_predictions_2017, x_pitchers,x_pitchers2017, pitcher_predictions, hitter_predictions):
    
    y_ERA = pitcher_predictions['correct_ERA'].tolist()
    y_K = pitcher_predictions['correct_K'].tolist()
    y_W = pitcher_predictions['correct_W'].tolist()
    y_WHIP = pitcher_predictions['correct_WHIP'].tolist()
    
    y_AVG = hitter_predictions['correct_AVG'].tolist()
    y_RBI = hitter_predictions['correct_RBI'].tolist()
    y_R = hitter_predictions['correct_R'].tolist()
    y_HR = hitter_predictions['correct_HR'].tolist()
    y_SB = hitter_predictions['correct_SB'].tolist()

    # lists of stats of interest
    hit_stats = ['AVG', 'HR', 'R', 'RBI', 'SB']
    pitch_stats = ['ERA', 'K', 'W', 'WHIP']
    all_stats = ['AVG', 'HR', 'R', 'RBI', 'SB', 'ERA', 'K', 'W', 'WHIP']
    
    # create dictionary of best params as learned in the machine learning portion of the project
    # key is the stat, value is the best parameter values
    best_params_all = {'correct_AVG': xgb_hitters_params[0]['correct_AVG'], \
                       'correct_HR': xgb_hitters_params[1]['correct_HR'],\
                       'correct_R': xgb_hitters_params[2]['correct_R'],'correct_RBI': xgb_hitters_params[3]['correct_RBI'], \
                       'correct_SB': xgb_hitters_params[4]['correct_SB'], \
                       'correct_ERA': rforest_pitchers_params[0]['correct_ERA'],\
                       'correct_K': xgb_pitchers_params[1]['correct_K'], \
                       'correct_W': rforest_pitchers_params[2]['correct_W'], \
                       'correct_WHIP': xgb_pitchers_params[3]['correct_WHIP']}
    # create dictionary of best model as learned in the machine learning portion of the project
    # key is the state, value is the best model
    models = {'AVG':'XGBoost', 'HR':'XGBoost', 'R':'XGBoost', 'RBI':'XGBoost', 'SB':'XGBoost', 'ERA':'Random Forest',\
              'K':'XGBoost', 'W':'Random Forest', 'WHIP':'XGBoost'}
    y_vals = {'AVG': y_AVG, 'HR': y_HR, 'R': y_R, 'RBI': y_RBI, 'SB': y_SB, \
              'ERA': y_ERA, 'K': y_K, 'W': y_W, 'WHIP': y_WHIP}
    
    # given a statistic and a player, return the best prediction for that player and that statistic
    def stat_pct(stat):
        """
        Inputs:
            stat (str): The statistic of interest - AVG, HR, R, RBI, SB for hitters, ERA, K, W, WHIP for pitchers
        Returns:
            vals ((# of players,3) ndarray): first column is player name, second column is predicted value for given
                                             statistic, third column is probability that the prediction is correct
        """
        # check that values given are valid
        if stat not in all_stats:
            print "Not an acceptable stat"
            return 'FAILED'
        
        if stat in hit_stats:
            # get a list of the names of all the hitters in the order that they appear in x_hitters2017
            name_list = hitter_predictions_2017['Name'].tolist()
            # run the model
            if models[stat] == 'XGBoost':
                # run XGBoost with best params for the stat
                xgbc = xgb(**best_params_all['correct_' + stat])
                model = xgbc.fit(x_hitters, y_vals[stat])
                preds = model.predict_proba(x_hitters2017)[:,1]
            if models[stat] == 'Random Forest':
                # run Random Forest with best params for the stat
                rf = RandomForestClassifier(**best_params_all['correct_' + stat])
                model = rf.fit(x_hitters, y_vals[stat])
                preds = model.predict_proba(x_hitters2017)[:,1]  
            # empty array to store names, stat predictions, and pct probability
            vals = np.empty((len(np.unique(name_list)), 3))
            # get list of unique names 
            unique_names = np.unique(name_list)
            # create lists to store percent probabilities and stat predictions
            pcts = np.zeros(len(unique_names))
            stats = np.zeros(len(unique_names))
            # loop through each player
            for j in xrange(len(unique_names)):
                # get indices of player
                idxs = []
                for x in xrange(len(name_list)):
                    if name_list[x] == unique_names[j]:
                        idxs.append(x)
                # find highest probability for given player, store the index and probability value
                vals_dict = dict((i, preds[i]) for i in idxs)
                b = collections.defaultdict(list)
                for key, value in vals_dict.iteritems():
                    b[value].append(key)
                pcts[j] = max(b.items())[0]
                #find corresponding value of stat
                stats[j] = hitter_predictions_2017[stat][max(b.items())[1][0]]
            vals[:,0] = stats
            vals[:,1] = pcts
            return vals, unique_names
                
        else:
            # get a list of the names of all pitchers in the order that they apear in x_pitchers2017
            name_list = pitcher_predictions_2017['Name'].tolist()
            # run the model
            if models[stat] == 'XGBoost':
                # run XGBoost with best params for the stat
                xgbc = xgb(**best_params_all['correct_' + stat])
                model = xgbc.fit(x_pitchers, y_vals[stat])
                preds = model.predict_proba(x_pitchers2017)[:,1]
            if models[stat] == 'Random Forest':
                # run Random Forest with best params for the stat
                rf = RandomForestClassifier(**best_params_all['correct_' + stat])
                model = rf.fit(x_pitchers, y_vals[stat])
                preds = model.predict_proba(x_pitchers2017)[:,1]
                
            # empty array to store names, stat predictions, and pct probability
            vals = np.empty((len(np.unique(name_list)), 3))
            # get list of unique names 
            unique_names = np.unique(name_list)
            # create lists to store percent probabilities and stat predictions
            pcts = np.zeros(len(unique_names))
            stats = np.zeros(len(unique_names))
            # loop through each player
            for j in xrange(len(unique_names)):
                # get indices of player
                idxs = []
                for x in xrange(len(name_list)):
                    if name_list[x] == unique_names[j]:
                        idxs.append(x)
                # find highest probability for given player, store the index and probability value
                vals_dict = dict((i, preds[i]) for i in idxs)
                b = collections.defaultdict(list)
                for key, value in vals_dict.iteritems():
                    b[value].append(key)
                pcts[j] = max(b.items())[0]
                #find corresponding value of stat
                stats[j] = pitcher_predictions_2017[stat][max(b.items())[1][0]]
            vals[:,0] = stats
            vals[:,1] = pcts
            return vals, unique_names
            
    # create empty master dataframes to hold all the hitter/pitcher predictions
    best_hitter_predictions = pd.DataFrame(columns = ['Name', 'AVG', 'AVG%', 'HR', 'HR%', 'R', 'R%', 'RBI', 'RBI%',\
                                                      'SB', 'SB%'])
    best_pitcher_predictions = pd.DataFrame(columns=['Name', 'K', 'K%', 'ERA', 'ERA%', 'W', 'W%', 'WHIP', 'WHIP%'])
    
    # fill the dataframes
    for stat in hit_stats:
        info = stat_pct(stat)
        best_hitter_predictions[stat] = info[0][:,0]
        best_hitter_predictions[stat+'%'] = info[0][:,1]
    best_hitter_predictions['Name'] = info[1]
    for stat in pitch_stats:
        info = stat_pct(stat)
        best_pitcher_predictions[stat] = info[0][:,0]
        best_pitcher_predictions[stat+'%'] = info[0][:,1]
    best_pitcher_predictions['Name'] = info[1]
    
    return best_hitter_predictions, best_pitcher_predictions
