'''

author: Chi Tang

1. In this note book, we generate a schdule to properly arrange the working time of nurses given their available time. 
2. The maximum working time per week for each people can be adjusted accordingly. The total time of each nurse's working
time cannot exceed the maximum working time.
3. The samples of each nurse's time schedule are generated randomly. By loading the real data, we can improve the algorithm
to fit the practical case.


'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pandas.tools.plotting import table
import seaborn
import json, re, os

#%matplotlib inline

def label_working_time_interval():

	"""
	Systematically label the working time interval from:
	8am to 6pm in a week. 

	The time interval in a day can be adjusted easily.

	"""
	time_schedule = []

	col = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri','Sat','Sun']   # Working time from Monday to Sunday 8am to 6pm
	index = [
	        '8:00 - 10:00',
	        '10:00 - 12:00',
	        '12:00 - 14:00',
	        '14:00 - 16:00',
	        '16:00 - 18:00'
	        ]

	for _ in range(1,6):
	    day_str = []
	    for day in col:
	        day_str.append(day + '_' + str(_))
	    time_schedule.append(day_str)

	time_intervals = pd.DataFrame(time_schedule, columns=col, index=index)

	#print self.time_intervals

	# Collect all the labelled time intervals

	timeslot_collection = np.ravel(time_intervals)
	#print timeslot_collection
	return time_intervals, timeslot_collection


def generate_nurse_data(time_intervals):

	"""
	
 	Each nurse will have their own available time schedule. 
 	They may not be able to work on some time intervals during the week.
 	Each nurse maximum hours per week can be adjusted based on real situations.

	"""

	NUM_NURSE = 10
	NUM_HOSPiTAL = 1
	nurse_name = []

	for _ in range(1,NUM_NURSE + 1):
	    nurse_name.append('nurse_' + str(_))

	#print "There are %d nurses to be arranged in %d hospital \n"%(NUM_NURSE, NUM_HOSPiTAL)
	#print nurse_name

	nurse_own_schedule = {}  # Create a nurse schedule to store the information.

	for nurse in nurse_name:
	    df_nurse = time_intervals.copy()
	    for day in range(7):
	        rand_time = np.random.randint(1,4)
	        for i in np.random.randint(0,5,rand_time):
	            df_nurse.iloc[i][day] = 'Not_available'
	    nurse_own_schedule[nurse] = df_nurse

	# save the information of all the randomly generated data.generated
	# in reality, it could be loaded from a csv file or other formats.    
	if not os.path.isdir('utils/'):
		os.mkdir('utils/')

	for key in nurse_own_schedule.keys():
	    nurse_own_schedule[key].to_csv('utils/'+ key + 'available_time.csv')
	    #print 'schedule of  ' + str(key)

	return nurse_name, nurse_own_schedule


def plot_table(printContext = True, saveFig = False, **kwargs):
     
    for key, cells in kwargs.iteritems():
        
        if printContext == True:
            print "The schedule of {} is:".format(key)


        colors = cells.applymap(lambda x: 'lightgray' if x== 'Not_available' else 'lightcoral') 

        fig = plt.figure(figsize=(8,4))

        ax = plt.subplot(2, 1, 1, frame_on = True)  # no visible frame
        #ax.xaxis.set_visible(False)  # hide the x axis
        #ax.yaxis.set_visible(False)
        ax.axis('off')

        tb1 = table(ax,cells,
                    loc='center',
                    cellLoc='center',
                    cellColours=colors.as_matrix(),
                    fontsize=14
              )

        plt.tight_layout()
       
        if saveFig == True:
            if not os.path.isdir('figs/'):
                os.mkdir('figs/')
            plt.savefig('figs/'+key+'.png', bbox_inches='tight', dpi = 150)
        
        # refresh the plot
        plt.show()
        plt.close()


def plot_arrangement(data, saveFig = False):

	### Plot the final arrangement

    print "The arrangement of the nurses in this week is:"


    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',\
                   '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',\
                   '#bcbd22', '#17becf']

    colors = data.applymap(lambda x: color_list[int(x[int(x.find('_'))+1:])-1])  # label each nurse to the specific color

    fig = plt.figure(figsize=(8,4))

    ax = plt.subplot(2, 1, 1, frame_on = True)  # no visible frame
    #ax.xaxis.set_visible(False)  # hide the x axis
    #ax.yaxis.set_visible(False)
    ax.axis('off')

    tb1 = table(ax,data,
                loc='center',
                cellLoc='center',
                cellColours=colors.as_matrix(),
                fontsize=14
          )

    plt.tight_layout()

    if saveFig == True:
        if not os.path.isdir('figs/'):
            os.mkdir('figs/')
        plt.savefig('figs/arrangement.png', bbox_inches='tight', dpi = 150)

    # refresh the plot
    plt.show()
    plt.close()


def create_available_nurse_map(nurse_name, timeslot_collection, nurse_own_schedule):

	# create a hashmap to store all the available nurses on each time slot
	# save the information 

	available_nurse_map = {key: [] for key in timeslot_collection}
	
	#print available_nurse_map

	for name in nurse_name:
	    
	    nurse_arr = np.ravel(nurse_own_schedule[name], order = 'F') 
	    nurse_arr = np.delete(nurse_arr, np.where(nurse_arr == 'Not_available'))
	    for i in nurse_arr:
	        available_nurse_map[i].append(name)

	#print "For example, on Monday from 8:00 to 10:00, the available nurses are: "
	#print available_nurse_map['Mon_1']
	return available_nurse_map


def main():

	"""
	Schedule a nurse on on each time interval. 

	Count the total time of each nurse. 

	Make sure the total time of the arranged nurse is below the maximum working hours.
	"""

	time_intervals, timeslot_collection = label_working_time_interval()
	nurse_name, nurse_own_schedule = generate_nurse_data(time_intervals)
	available_nurse_map = create_available_nurse_map(nurse_name, timeslot_collection, nurse_own_schedule)

	#plot_table(printContext = True, saveFig = True, **nurse_own_schedule)

	arrangement = pd.DataFrame(data = None, index = time_intervals.index, columns= time_intervals.columns)


	# Initialize the hours of each nurse to zero
	nurse_hours = {key: 0 for key in nurse_name}
	MAX_HOURS = 10 # the maximum working hours is set to 10 hours here.

	available_nurse_map_copy = available_nurse_map.copy() 

	for each_time in timeslot_collection:   
	    # parse the each_time into index and column number in the data frame
	    col, idx = each_time[:each_time.find('_')], int(each_time[each_time.find('_')+1:])
	    #print idx,col
	    
	    while True:
	        if not available_nurse_map_copy[each_time]: 
	            arrangement[col][idx-1] # pay attention to the idx here.
	            break
	        else:
	            select_nurse = np.random.choice(available_nurse_map_copy[each_time])
	             # if the working hours of selected nurse is over 10, then drop the selection in the list.
	            if nurse_hours[select_nurse] >= 10: 
	                available_nurse_map_copy[each_time].remove(select_nurse)
	            #otherwise, arrange the selected nurse to work on this time interval.
	            # make sure to update their work time.
	            else:
	                nurse_hours[select_nurse] += 2  # The working hours for each time slot.
	                arrangement[col][idx - 1] = select_nurse
	                break
                


	plot_arrangement(data = arrangement, saveFig = True)

	# save the information
	if not os.path.isdir('utils/'):
	    os.mkdir('utils/')
	    
	with open("utils/available_nurse_map.txt",'w') as f:
	    json.dump(available_nurse_map, f)


if __name__ == '__main__':

	main()

