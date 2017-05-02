from __future__ import division
from matplotlib import pyplot as plt
from operator import truediv
import pandas as pd
import re
import numpy as np
import os


def stat_plot(df, season, statistic,method = 'compare'):
    """this function will plot the predicted and the actual sum of a given statistic for each season 
    parameters:
    df (pandas dataframe): the prediction method's dataframe that we want to graph
    season (list): List of seasons that we want to plot for
    statistic (string): the statistic we want to plot
    method: determine whether to graph projected and actual side by side (method==compare) or graph projected/actual
    returns: nothing, but it does make a bar graph of the aggregated stats for the actual and projected """
    x_list = [] #list for projected stats
    x2_list = [] #list for actual stats
    for i in season:
        df_subset = df[df['Season']==i]
        x_list.append(df_subset[statistic].sum())
        x2_list.append(df_subset['actual_'+statistic].sum())
        
    x3_list = map(truediv, x_list, x2_list) #list of proportions of projected/actual per year
    #set up the dimensions of the graph
    index = np.arange(len(season))
    bar_width = 0.35
    if method == 'compare':
        #makes bars for the projected stats
        rects1 = plt.bar(index, x_list, bar_width,
                         #color='b',
                         label='Projected')
        #Makes bars for the actual stats
        rects2 = plt.bar(index + bar_width, x2_list, bar_width,
                         #color='g',
                         label='Actual')
        plt.xlabel('Season')
        plt.ylabel(statistic)
        #plt.title(statistic + ' per season')
        plt.xticks(index + bar_width/2, season)
    plt.legend(loc = 'best')

def aggregate_bar_graphs_hitters(guru_hitters, fangraphs_hitters, steamer_hitters, espn_hitters, cbs_hitters, marcel_hitters):
	plt.figure(figsize=(18,10))
	#Sets up the subplots for our 6 graphs using the stat_plot method
	plt.subplot(231)
	stat_plot(guru_hitters,[2010,2011, 2012, 2013, 2014, 2016],'R' )
	plt.ylim(0,25000)
	plt.title("GURU Runs/season")

	plt.subplot(232)
	stat_plot(fangraphs_hitters,[2010,2011, 2012, 2013, 2014, 2015], "R" )
	plt.ylim(0,25000)
	plt.title("Fangraphs Runs/season")


	plt.subplot(233)
	stat_plot(steamer_hitters, [2010,2011, 2012, 2013, 2014, 2015],"R")
	plt.ylim(0,25000)
	plt.title("Steamer Runs/season")


	plt.subplot(234)
	stat_plot(espn_hitters, [2011, 2012, 2013, 2014, 2015],"R")
	plt.ylim(0,25000)
	plt.title("ESPN Runs/season")


	plt.subplot(235)
	stat_plot(cbs_hitters, [2010,2011, 2012, 2013, 2014],"R")
	plt.ylim(0,25000)
	plt.title("CBS Runs/season")

	plt.subplot(236)
	stat_plot(marcel_hitters, [2010,2011, 2012, 2013, 2014, 2015],"R")
	plt.ylim(0,25000)
	plt.title("Marcel Runs/season")

	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	plt.show()

def aggregate_bar_graphs_pitchers(guru_pitchers, fangraphs_pitchers, steamer_pitchers, espn_pitchers, cbs_pitchers, marcel_pitchers):
	#Now we will do a similar graph as above only we will compare projections of strikeouts to actual
	plt.figure(figsize=(18,10))

	#Sets up the subplots for our 6 graphs using the stat_plot method
	plt.subplot(231)
	stat_plot(guru_pitchers,[2010,2011, 2012, 2013, 2014, 2016],'K' )
	plt.ylim(0,40000)
	plt.title("GURU K's/season")

	plt.subplot(232)
	stat_plot(fangraphs_pitchers,[2010,2011, 2012, 2013, 2014, 2015], "K" )
	plt.ylim(0,40000)
	plt.title("Fangraphs K's/season")

	plt.subplot(233)
	stat_plot(steamer_pitchers, [2010,2011, 2012, 2013, 2014, 2015],"K")
	plt.ylim(0,40000)
	plt.title("Steamer K's/season")

	plt.subplot(234)
	stat_plot(espn_pitchers, [2011, 2012, 2013, 2014, 2015],"K")
	plt.ylim(0,40000)
	plt.title("ESPN K's/season")

	plt.subplot(235)
	stat_plot(cbs_pitchers, [2011, 2012, 2013, 2014, 2015],"K") 
	plt.ylim(0,40000)
	plt.title("CBS K's/season")

	plt.subplot(236)
	stat_plot(marcel_pitchers, [2010,2011, 2012, 2013, 2014, 2015],"K")
	plt.ylim(0,40000)
	plt.title("Marcel K's/season")

	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	plt.show()
