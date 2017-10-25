# Nurse scheduling problem

## Nurse_schedule.ipynb
scheduling code in jupyter notebook

## Nurse_schedule.py
scheduling code in python file

## figs
contain output figs of arrangement

## utils
randomly generated nurse schedule data and save few processed data

## Nurse scheduling problem 

## Implementing linear programming  algorithm to solve the NP-hard nurse scheduling problem. 


## Now more general cases:

* Each day is divided into multiple shifts(usually 2-4).

* There are a number of required nurses for each of the shift. 

* A nurse is off on a specific day if no shift is assigned or if the nurse has requested a PTO on that specific day. 

* The planning length is flexible (test case using 4 weeks). 
    
* Each nurse should work in range(max_nurse_shifts), let's say max 5 shifts per week.

* Nurse will only work at most one shift per day.

* A nurse who works on a late nigth shift will take the next day off.

* Max numbers of night shift for each nurse per week is at most ONE.


## Model:

* Essential idea is to introduce a binary variable ***x_ns*** in order to linearize the model

* Constraints and objective function can be represented as equality and inequality equations.

\begin{align}
    x_{ns}= 
        \begin{cases}
            0, &\text{nurse n will not work on shift s}\ \\
            1, &\text{nurse n will work on shift s}\
        \end{cases}
\end{align}

* s: each shift

* n: each nurse

* r: list storing the required nurses in each day.


\begin{align}
    \text{r} = [\text{number of nurses required in specific shift s}] \\
    \text{r[even index]} = [\text{number of nurses required in day shifts}] \\
    \text{r[odd index]} = [\text{number of nurses required in night shifts}] 
\end{align}

* PTO = Dictionary which stores the information of the off-work shifts(nurses requested PTO or other reasons)

\begin{align}
    \text{PTO} = \{n : [\text{list of off-work shifts for nurse n}]\}
\end{align}


* daily_shift: [0, 1, 2, ...] in this case. The last element is late night shift.

* working_shift = [1,..., max_nurse_shifts]: nurse works either 8, 16, 24, 32, 40 hours per week.

* planning_length: in this notebook, planning_length is flexible. 

* Objective: minimize the numbers of scheduled nurses after satisfying all constraints. 


* Implementing linear programming with pulp python package to find the solution of this constrained optimization problem. 