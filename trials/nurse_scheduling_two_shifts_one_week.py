'''

Author: Chi Tang

# Nurse scheduling problem 

## Implementing linear programming  algorithm to solve the NP-hard nurse scheduling problem. 


### Some basic constraints and assumptions:

* Each day is divided into 2 separate shifts of 12 hours (day/night).

* There are a number of required nurses for each of them. 

* A nurse is off on a specific day if no shift is assigned or if the nurse has requested a PTO on that specific day. 

* The planning length is 4 weeks (28 days). 
    
    * In this notebook, used 1 week instead for the purpose of convenience and easy visualization.

* Each nurse should work either 12, 24 or 36 hours hours per week.


### I personally add few more constraints:

* No nurse will work on both shifts in the single day.

* A nurse who works on a nigth shift will take the next day off.

* Max numbers of night shift for each nurse is at most ONE.

'''

import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools.plotting import table
import os, sys, json
from pulp import *


class nurse:

	"""
	nurse class for the scheduling problem
	"""

	def __init__(self, nurse_max_shifts = 3, planning_length = 1, daily_shift = 2, 
		nurse_per_shift = None, total_nurses = None):

		# nurse works either 12, 24, 36 hours per week.
		self.nurse_working_shift = range(1, nurse_max_shifts + 1) 
		self.nurse_max_shifts = nurse_max_shifts

		# planning_length: 1 week, 2 week or 4 weeks
		self.n = planning_length

		# numbers of shift per day
		# for example: 
		# [day, night] = [0, 1]
		self.daily_shift = range(daily_shift)

		# label each day from Monday to Sunday:
		self.day = ['M', 'Tu', 'W', 'Th', 'F','Sa','Su']
		self.shift_name = []
		for w in range(1, self.n+1):
			for d in self.day:
				for i in self.daily_shift:
					self.shift_name.append('week'+str(w)+'_'+str(d)+'_'+str(i))

		"""
		Create the require_nurses for all the shifts.
		Basically, the length of shift list is:
		len(shifts) = daily_shift * 7 * planning_length
		
		for example: if daily_shift is 2 (day/night), planning_length is 1
		shifts[0] = First week Monday day shift
		shifts[1] = First week Monday night shift
		shifts[2] = First week Tuesday day shift
		...
		shifts[-1] = Last week Sunday night shift

		"""
		self.shifts = range(daily_shift * 7 * planning_length)
		
		"""
		required nurses for shifts are:
		randomly generated value if not provided (for demo purpose)
		
		for example, day shift needs 5 nurses each day, night shift needs 3 nurses
		the corresponding required nurses will be:
		[5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3]
		"""
		if not nurse_per_shift:
			self.nurse_per_shift = [5, 3]
		else:
			self.nurse_per_shift = nurse_per_shift

		self.r = self.nurse_per_shift * 7 * planning_length


		# total nurses working in the hospital
		if not total_nurses:
			self.total_nurses = int(sum(self.r)/2)
		else:
			self.total_nurses = total_nurses

		# create nurses list and nurses_id tag:
		# nurses: label each nurse. Simply use integers to represent.
		self.nurses = range(self.total_nurses)
		self.nurses_id = ['nurse'+str(i) for i in range(self.total_nurses)]


		"""
		create a off_shift dictionary for the nurses who requested PTO or
		have a shift off.
		"""
		self.off_shift = {}

		# initialize a linear programming problem
		self.prob = LpProblem("Nurse scheduling",LpMinimize)

	
	def update_off_shift(self, fileName = 'utils/off_shift.json'):

		'''
		Update the off_shift in the json file.
		'''

		try:
			with open(fileName, 'r') as fp:
				self.off_shift = json.load(fp)
		except IOError:
			print "The file path does not exist"
			sys.exit(-1)

		print self.off_shift


	def lp_problem(self):

	    '''
	    Use pulp to solve the constrained problem using linear programming(LP) algorithm. 
	    1. Create LpVariables. Binary category in this case
	    2. Add constraints in either equality or inequality conditions.
	    3. Building objective using LpObjective. 
	    '''
	    
	    # Creating the variables. 
	    self.var = {
	     (n, s): LpVariable(
	        "schdule_{0}_{1}".format(n, s), cat = "Binary")
	        for n in self.nurses for s in self.shifts
	    }

	    
	    # add constraints: 
	    # Nurses do not work in two consecutive shifts
	    # If nurse works for a night shift, he/she will take a next day off

	    for n in self.nurses:
	        for s in self.shifts:
	            if s%2 == 0:
	                self.prob.addConstraint(
	                self.var[(n,s)] + self.var[(n, s+1)] <= 1  # for day shift
	               )
	            elif s%2 == 1 and s < self.shifts[-1]:
	                # night shift. Do not forget to add condition that the last
	                # shift in the scheduling does not count.
	                self.prob.addConstraint(
	                self.var[(n, s)] + self.var[(n, s+1)] + self.var[(n, s+2)] <= 1
	                )

	    # add constraints:
	    # Request PTO or take a specific day off:
	    for n in self.nurses:
	        if str(n) in self.off_shift:
	            for s in self.off_shift[str(n)]:
	                self.prob.addConstraint(
	                self.var[(n, s)] == 0
	                )

	    # add constraints:
	    # Working shift is either 1, 2 or 3
	    # Here the first calculation is based on 
	    for n in self.nurses:
	        self.prob.addConstraint(
	        sum(self.var[(n,s)] for s in self.shifts) <= self.nurse_max_shifts    
	        )
	        self.prob.addConstraint(
	        sum(self.var[(n,s)] for s in self.shifts) >= 1
	        )

	    # add constraints
	    # Max numbers of night shift is one for each nurse

	    for n in self.nurses:
	        self.prob.addConstraint(
	        sum(self.var[(n, s)] for s in self.shifts if s%2) <= 1
	        )
	    # add constraints
	    # for each shift, the numbers of working nurses should be greater than
	    # the required numbers of nurses
	    for s in self.shifts:
	        try:
	            self.prob.addConstraint(
	            sum(self.var[(n,s)] for n in self.nurses) >= self.r[s]
	            )
	        except:
	            print "len(shifts) should be equal to len(require_nurses)"
	            sys.exit(-1)   

	            
	    # add objective: minimize the numbers of total nurses required
	    # nurse_working = []
	    # for s in shifts:
	    #     nurse_shift = sum(var[(n, s)] for n in nurses)
	    #     nurse_working.append(
	    #     pulp.LpVariable("nurses_%d"%(s,), cat = 'Integer', lowBound = 0)
	    #     )
	    self.prob.objective = sum(self.var[(n,s)] for n in self.nurses for s in self.shifts)   

	    return self.prob
	    

	def lp_solve(self, solver = None):

		# problem solver

		# if solver:
		# 	self.prob.solve(solver)
		# else:
		# 	self.prob.solve()
		self.prob.solve()


		print "The status of solving the problem is: "
		print LpStatus[self.prob.status]


	def nurse_scheduling(self):

		# output the whole scheduling

		self.sch = pd.DataFrame(data=None, index = self.nurses_id, columns = self.shift_name) 
		for k, v in self.var.items():
			n, s = k[0], k[1]
			self.sch.iloc[n][s] = int(value(v))

		return self.sch


	def schedule_which_nurse(self, nurseWho = 0):
	    
	    '''
	    nurseWho: nurse id
	    table: Either the schedule dataframe or the linear programming solution
	    inputType: if 'lp': use linear programming solution
	               else: the dataframe
	    '''
	    # Get the data for scheduling nurse n:
	    res = []
	       
	    for s in self.shifts:
	        res.append(
	            int(value(self.var[(nurseWho, s)]))
	            )
	    
	    num_shift = len(self.daily_shift)
	    res = np.array(res).reshape(len(res)/num_shift, num_shift).swapaxes(0, 1)
	    
	    col = ['week'+str(w) + '_' + str(d) for w in range(1, self.n + 1) for d in self.day]
	    df_sch = pd.DataFrame(res, index = self.daily_shift, columns = col)
	    
	    return df_sch


	def check_off_shift(self):
		
		# return True if the constraints have been satisfied otherwise return False

	    for n in self.nurses:
	        if str(n) in self.off_shift:
	            for s in self.off_shift[str(n)]:
	                if value(self.var[(n,s)]) == 1:
	                	print n, s
	                	return False

	    for n in self.nurses:
	        for s in self.shifts:
	            if s%2 == 0:
	                if value(self.var[(n,s)]) + value(self.var[(n, s+1)]) > 1:
	                	return False  # for day shift
	            elif s%2 == 1 and s < self.shifts[-1]:
	                if value(self.var[(n,s)]) + value(self.var[(n,s+1)]) + value(self.var[(n, s+2)]) > 1:
	                	return False

	    for n in self.nurses:
	        tmp = sum(value(self.var[(n,s)]) for s in self.shifts) 
	        if tmp > self.nurse_max_shifts or tmp < 1:
	        	return False

	    for n in self.nurses:
	        if sum(value(self.var[(n, s)]) for s in self.shifts if s%2) > 1:
	        	return False

	    for s in self.shifts:
	        try:
	            if sum(value(self.var[(n,s)]) for n in self.nurses) < self.r[s]:
	            	return False
	        except:
	            print "len(shifts) should be equal to len(require_nurses)"
	            sys.exit(-1)   

	    return True
	                 


def plot_table(df, figSize = (4,2), saveFig = False, figTitle = 'nurse_scheduling'):
     
    # visulize the schedule  
    colors = df.applymap(lambda x: 'lightgray' if x== 0 else 'lightcoral') 

    fig = plt.figure(figsize=figSize)

    ax = plt.subplot(2, 1, 1, frame_on = True)  # no visible frame
    #ax.xaxis.set_visible(False)  # hide the x axis
    #ax.yaxis.set_visible(False)
    ax.axis('off')

    tb1 = table(ax,df,
                loc='center',
                cellLoc='center',
                cellColours=colors.as_matrix(),
                fontsize=14
          )

    if saveFig == True:
        if not os.path.isdir('figs/2shifts1week/'):
            os.mkdir('figs/2shifts1week/')
        plt.savefig('figs/2shifts1week/'+ figTitle +'.png', bbox_inches='tight', dpi = 150)

    # refresh the plot
    #plt.show()
    plt.close()



if __name__ == '__main__':

	model = nurse()

	model.update_off_shift()

	prob = model.lp_problem()

	model.lp_solve()

	schedule = model.nurse_scheduling()

	print schedule.head(5)

	print "\nThe schedule of all nurses: "
	plot_table(schedule, figSize = (10, 4), saveFig = True)

	for n in model.nurses[:]:
		n0 = model.schedule_which_nurse(n)
		#print n0
	 	plot_table(n0, saveFig = True, figTitle = 'nurse' + str(n))


	print "\nCheck off_shift constraint: {}".format(model.check_off_shift())



