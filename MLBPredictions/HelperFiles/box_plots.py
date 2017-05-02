from __future__ import division
from matplotlib import pyplot as plt
from operator import truediv
import pandas as pd
import re
import numpy as np
import os

def box_plots_2016(hittersDF, pitchersDF):
	# boxplots for 2016

	# NOTE: Stats are split up into different graphs to deal with y-axis scaling
	plt.figure(figsize=(13,3))

	plt.subplot(1,3,1)
	hittersDF[hittersDF['Season'] == 2016].boxplot(column = ['actual_R', 'actual_HR', 'actual_RBI'])
	plt.ylabel('Number of R, HR, and RBI, respectively')
	plt.subplot(1,3,2)
	hittersDF[hittersDF['Season'] == 2016].boxplot(column = ['actual_SB'])
	plt.ylabel('Number of SB')
	plt.title('Actual Hitter Statistics for 2016')
	plt.subplot(1,3,3)
	hittersDF[hittersDF['Season'] == 2016].boxplot(column = ['actual_AVG'])
	plt.ylabel('BA')
	plt.show()

	plt.figure(figsize=(13,3))

	plt.subplot(1,3,1)
	pitchersDF[pitchersDF['Season'] == 2016].boxplot(column = ['actual_ERA', 'actual_WHIP'])
	plt.ylabel('ERA and WHIP, respectively')
	plt.subplot(1,3,2)
	pitchersDF[pitchersDF['Season'] == 2016].boxplot(column = ['actual_SV', 'actual_W'])
	plt.ylabel('Number of SV and W, respectively')
	plt.title('Actual Pitcher Statistics for 2016')
	plt.subplot(1,3,3)
	pitchersDF[pitchersDF['Season'] == 2016].boxplot(column = 'actual_K')
	plt.ylabel('Number of SO')
	plt.show()

def box_plots_avg(guru_hitters, cbs_hitters, espn_hitters, steamer_hitters, fangraphs_hitters, marcel_hitters, hittersDF):
	# 2014 projections for strikeouts

	plt.figure(figsize=(15,9))

	plt.subplot(3,3,1)
	guru_hitters[guru_hitters['Season']==2014].boxplot(column='AVG')
	plt.ylim(.15,.40)
	plt.ylabel('Batting Average (AVG)')
	plt.title("Guru AVG Projections")

	plt.subplot(3,3,2)
	cbs_hitters[cbs_hitters['Season']==2014].boxplot(column='AVG')
	plt.ylim(.15,.40)
	plt.title("CBS AVG Projections")

	# FIX THIS ------------------------------------------------------------------------
	plt.subplot(3,3,3)
	espn_hitters[espn_hitters['Season']==2014].boxplot(column='AVG')
	plt.ylim(.15,.40)
	plt.title("ESPN AVG Projections")


	plt.subplot(3,3,4)
	steamer_hitters[steamer_hitters['Season']==2014].boxplot(column='AVG')
	plt.ylim(.15,.40)
	plt.ylabel('Batting Average (AVG)')
	plt.title("Steamer AVG Projections")

	plt.subplot(3,3,5)
	fangraphs_hitters[fangraphs_hitters['Season']==2014].boxplot(column='AVG')
	plt.ylim(.15,.40)
	plt.title("FanGraphs AVG Projections")

	plt.subplot(3,3,6)
	marcel_hitters[marcel_hitters['Season']==2014].boxplot(column='AVG')
	plt.ylim(.15,.40)
	plt.title("Marcel AVG Projections")

	plt.subplot(3,3,7)
	hittersDF[hittersDF['Season']==2014].boxplot(column='actual_AVG')
	plt.ylim(.15,.40)
	plt.title("Actual AVG Values")
	plt.ylabel('Batting Average (AVG)')
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	plt.show()
