from __future__ import division
from matplotlib import pyplot as plt
from operator import truediv
import pandas as pd
import re
import numpy as np
import os


def correct_lists(df,stat_dict):
    """This function creates various lists of correct predictions, under predictions, and over predictions
    for a given set of dictionaries and a given epsilon ball. Also, because we are using an epsilon ball we will have a 
    number of predictions that are classified as 'correct' but we will still be over or under the actual value. We will
    use these lists in future analysis, so we will also record these values. 
    Parameters:
    df (dataframe): pass in one of our prediction dataframes that has both the predictions and the real stats
    stat_dict (dictionary): pass in a dictionary with the keys as the statistics and the values as the epsilon ball around
                            the statistic that we will consider a 'correct' prediction.
    returns:
    stat_list (list): list of ratios of #correct/(total # of predictions) for each statistic
    now for each 'correct' prediction they can still be over, under, or = to the actual value so we record lists for each
    number_under (list): gives a list of ratios of the #underpredicted of the correct predictions/total number of correct
                        predictions
    number_exact (list): gives a list of ratios of the #exactly correct predictions/total number of correct
                        predictions
    number_over (list): gives a list of ratios of the #over correct predictions/total number of correct
                        predictions"""
    #set up our lists
    stat_list = [] 
    number_under =[]
    number_exact =[]
    number_over = []
    total_pred = df.shape[0]
    #run through each stat in the dictionary to find the values for each list
    for i in stat_dict:
        #define the epsilon ball from the values of the dictionary
        difference = stat_dict.get(i)
        
        #Set up a new df which is all the correct predictions from the passed in df
        correct = df[(df['actual_'+i]<df[i]+difference)&(df['actual_'+i]>df[i]-difference)]
        number_correct = correct.shape[0]
        
        #now we take the size of the subsets of the df that correspond to under, over, exact
        under = correct[correct['actual_'+i]>correct[i]].shape[0]
        over = correct[correct['actual_'+i]<correct[i]].shape[0]
        exact = correct[correct['actual_'+i]==correct[i]].shape[0]
        
        number_under.append(float(under)/number_correct)
        number_over.append(float(over)/number_correct)
        number_exact.append(float(exact)/number_correct)
        stat_list.append(float(number_correct)/total_pred)
    return stat_list, number_under, number_over, number_exact



def plot_correct(steamer_pitchers, fangraphs_pitchers, cbs_pitchers, marcel_pitchers, espn_pitchers, guru_pitchers,\
                steamer_hitters, fangraphs_hitters, cbs_hitters, marcel_hitters, espn_hitters, guru_hitters,position='hitters'):
    """This function makes a bar graph. The function is past in a list of values from each projection method
    and then plots the graphs.
    parameters:
    method (list of lists): This contains a list of every projection methods correct ratio lists for the given statistics
    position (string): This is to determine whether the projection methods are hitters or pitchers """
    plt.figure(figsize=(13,5.5))
    
    #Set up the dimensions for each bar
    #index = np.arange(0, len(method[0]) * 2, 2)
    #index = np.arange(0, 6 * 2, 2)
    bar_width = 0.25
    
    #Set up the labels for the x axis
    if position == 'hitters':
        steamer = correct_lists(steamer_hitters,{'RBI':7,'AVG':.01,'R':7, 'HR':5, 'SB':3})[0]
        fangraphs = correct_lists(fangraphs_hitters,{'RBI':7,'AVG':.01,'R':7, 'HR':5, 'SB':3})[0]
        cbs = correct_lists(cbs_hitters,{'RBI':7,'AVG':.01,'R':7, 'HR':5, 'SB':3})[0]
        marcel = correct_lists(marcel_hitters,{'RBI':7,'AVG':.01,'R':7, 'HR':5, 'SB':3})[0]
        espn = correct_lists(espn_hitters,{'RBI':7,'AVG':.01,'R':7, 'HR':5, 'SB':3})[0]
        guru = correct_lists(guru_hitters,{'RBI':7,'AVG':.01,'R':7, 'HR':5, 'SB':3})[0]
        method = [steamer, fangraphs, cbs, marcel, espn, guru]
        #statistic = ['RBI','AVG','R','HR','SB']
        statistic = ['SB','HR','R','AVG','RBI']
        index = np.arange(0, len(method[0]) * 2, 2)
    else:
        steamer = correct_lists(steamer_pitchers,{'W':2,'K':15,'ERA':.2, 'WHIP':.05})[0]
        fangraphs = correct_lists(fangraphs_pitchers,{'W':2,'K':15,'ERA':.2, 'WHIP':.05})[0]
        cbs= correct_lists(cbs_pitchers,{'W':2,'K':15,'ERA':.2, 'WHIP':.05})[0]
        marcel = correct_lists(marcel_pitchers,{'W':2,'K':15,'ERA':.2, 'WHIP':.05})[0]
        espn = correct_lists(espn_pitchers,{'W':2,'K':15,'ERA':.2, 'WHIP':.05})[0]
        guru = correct_lists(guru_pitchers,{'W':2,'K':15,'ERA':.2, 'WHIP':.05})[0]
        method = [steamer,fangraphs, cbs, marcel, espn, guru]
        statistic = ['K', 'WHIP', 'ERA', 'W', 'S']
        index = np.arange(0, len(method[0]) * 2, 2)

    #set up our different bars 
    rects1 = plt.bar(index, method[0], bar_width,label='Steamer')
    rects2 = plt.bar(index + bar_width, method[1], bar_width,label='Fangraphs')
    rects3 = plt.bar(index + 2*bar_width, method[2], bar_width,label='CBS')
    rects4 = plt.bar(index + 3*bar_width, method[3], bar_width,label='Marcel')
    rects5 = plt.bar(index + 4*bar_width, method[4], bar_width,label='ESPN')
    rects6 = plt.bar(index + 5*bar_width, method[5], bar_width,label='GURU')

    plt.xlabel('Statistic')
    plt.ylabel('Percent correct')
    plt.title('Correct Predictions Comparisons')
    plt.xticks(index + 2.5*bar_width, statistic)
    plt.legend()
    plt.show()


def plot_ratio(df, stat_dict, position = 'hitters'):
    projection, under, over, exact = correct_lists(df,stat_dict)
    statistics = stat_dict.keys()
    if position == 'pitchers':
        index = np.arange(0, 4 * 2, 2)
    else:
        index = np.arange(0, 5 * 2, 2)
    bar_width = 0.35
    rects1 = plt.bar(index, under, bar_width,label='Under Predictions')
    rects2 = plt.bar(index + bar_width, exact, bar_width,label='Exact Predictions')
    rects3 = plt.bar(index + 2*bar_width, over, bar_width,label='Over Predictions')
    plt.xlabel('Statistic')
    plt.ylabel('Percent')
    plt.title('Correct Predictions Comparisons')
    plt.xticks(index + bar_width, statistics)
    plt.legend()

def hitters_ratio_plot(steamer_hitters, fangraphs_hitters, cbs_hitters, marcel_hitters, espn_hitters, guru_hitters):
    #Plot each projection methods ratio of under, over, and exact correct predictions for batters
    plt.figure(figsize=(13,7))
    plt.subplot(231)
    plot_ratio(steamer_hitters, {'RBI':7,'AVG':.005,'R':5, 'HR':5, 'SB':3})
    plt.ylim(0,.75)
    plt.title('Steamer batters correct prediction breakdown')

    plt.subplot(232)
    plot_ratio(fangraphs_hitters, {'RBI':7,'AVG':.005,'R':5, 'HR':5, 'SB':3})
    plt.ylim(0,.75)
    plt.title('Fangraphs batters correct prediction breakdown')

    plt.subplot(233)
    plot_ratio(cbs_hitters, {'RBI':7,'AVG':.005,'R':5, 'HR':5, 'SB':3})
    plt.ylim(0,.75)
    plt.title('CBS batters correct prediction breakdown')

    plt.subplot(234)
    plot_ratio(marcel_hitters, {'RBI':7,'AVG':.005,'R':5, 'HR':5, 'SB':3})
    plt.ylim(0,.75)
    plt.title('Marcel batters correct prediction breakdown')

    plt.subplot(235)
    plot_ratio(espn_hitters, {'RBI':7,'AVG':.005,'R':5, 'HR':5, 'SB':3})
    plt.ylim(0,.75)
    plt.title('ESPN batters correct prediction breakdown')

    plt.subplot(236)
    plot_ratio(guru_hitters, {'RBI':7,'AVG':.005,'R':5, 'HR':5, 'SB':3})
    plt.ylim(0,.75)
    plt.title('Guru batters correct prediction breakdown')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.legend(loc = 'best')
    plt.show()

def pitchers_ratio_plot(steamer_pitchers, fangraphs_pitchers, cbs_pitchers, marcel_pitchers, espn_pitchers, guru_pitchers):
    #Plot each projection methods ratio of under, over, and exact correct predictions for pitchers 
    plt.figure(figsize=(13,7))
    plt.subplot(231)
    plot_ratio(steamer_pitchers, {'W':2,'K':15,'ERA':.2, 'WHIP':.05}, 'pitchers')
    plt.ylim(0,.75)
    plt.title('Steamer pitchers correct prediction breakdown')

    plt.subplot(232)
    plot_ratio(fangraphs_pitchers, {'W':2,'K':15,'ERA':.2, 'WHIP':.05}, 'pitchers')
    plt.ylim(0,.75)
    plt.title('Fangraphs pitchers correct prediction breakdown')

    plt.subplot(233)
    plot_ratio(cbs_pitchers, {'W':2,'K':15,'ERA':.2, 'WHIP':.05}, 'pitchers')
    plt.ylim(0,.75)
    plt.title('CBS pitchers correct prediction breakdown')

    plt.subplot(234)
    plot_ratio(marcel_pitchers, {'W':2,'K':15,'ERA':.2, 'WHIP':.05}, 'pitchers')
    plt.ylim(0,.75)
    plt.title('Marcel pitchers correct prediction breakdown')

    plt.subplot(235)
    plot_ratio(espn_pitchers, {'W':2,'K':15,'ERA':.2, 'WHIP':.05}, 'pitchers')
    plt.ylim(0,.75)
    plt.title('ESPN pitchers correct prediction breakdown')

    plt.subplot(236)
    plot_ratio(guru_pitchers, {'W':2,'K':15,'ERA':.2, 'WHIP':.05}, 'pitchers')
    plt.ylim(0,.75)
    plt.title('Guru pitchers correct prediction breakdown')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.legend(loc = 'best')
    plt.show()