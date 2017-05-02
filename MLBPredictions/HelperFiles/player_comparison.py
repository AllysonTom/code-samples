from __future__ import division
from matplotlib import pyplot as plt
from operator import truediv
import pandas as pd
import re
import numpy as np
import os
def player_comparison(hittersDF):
	# grab SB statistics for each player of interest
	# taking 2009 off because when we made these graphs we did not have 2009 stats
	pence_SB_stats = hittersDF[(hittersDF['Name'] == 'hunter pence') & (hittersDF['Season'] != 2009)]\
	    ['actual_SB'].tolist()[::-1]
	gonzo_SB_stats = hittersDF[(hittersDF['Name'] == 'adrian gonzalez') & (hittersDF['Season'] != 2009)]\
	    ['actual_SB'].tolist()[::-1]
	ramirez_SB_stats = hittersDF[(hittersDF['Name'] == 'hanley ramirez') & (hittersDF['Season'] != 2009)]\
	    ['actual_SB'].tolist()[::-1]
	# grab HR statistics for each player of interest
	pence_HR_stats = hittersDF[(hittersDF['Name'] == 'hunter pence') & (hittersDF['Season'] != 2009)]\
	    ['actual_HR'].tolist()[::-1]
	gonzo_HR_stats = hittersDF[(hittersDF['Name'] == 'adrian gonzalez') & (hittersDF['Season'] != 2009)]\
	    ['actual_HR'].tolist()[::-1]
	ramirez_HR_stats = hittersDF[(hittersDF['Name'] == 'hanley ramirez') & (hittersDF['Season'] != 2009)]\
	    ['actual_HR'].tolist()[::-1]

	# list of years
	yrs = np.array([10, 11, 12, 13, 14, 15, 16])

	plt.figure(figsize=(9,3))

	plt.subplot(1,2,1)
	plt.plot(yrs, pence_SB_stats, label='Pence')
	plt.plot(yrs, gonzo_SB_stats, label='Gonzalez')
	plt.plot(yrs, ramirez_SB_stats, label='Ramirez')
	plt.title("Stolen Bases by Year")
	plt.legend(loc='best')
	plt.ylabel('Number of Stolen Bases')
	plt.xlabel('Season, 20--')
	plt.xlim(10, 16)

	plt.subplot(1,2,2)
	plt.plot(yrs, pence_HR_stats)
	plt.plot(yrs, gonzo_HR_stats)
	plt.plot(yrs, ramirez_HR_stats)
	plt.title("Home Runs by Year")
	plt.ylabel('Number of Home Runs')
	plt.xlabel('Season, 20--')
	plt.xlim(10, 16)
	plt.show()
