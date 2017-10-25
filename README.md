# Nurse scheduling problem

## Implementing linear programming  algorithm to solve the NP-hard nurse scheduling problem. 


## General constraints and assumptions:

* Each day is divided into multiple shifts(usually 2-4).

* There are a number of required nurses for each of the shift. 

* A nurse is off on a specific day if no shift is assigned or if the nurse has requested a PTO on that specific day. 

* The planning length is flexible (1-4 weeks, or more). 
    
* Each nurse should work in range(max_nurse_shifts), let's say max 5 shifts per week.

* Nurse will only work at most one shift per day.

* A nurse who works on a late nigth shift will take the next day off.

* Max numbers of night shift for each nurse per week is at most ONE.


## Model:

* Essential idea is to introduce a binary variable x_ns in order to linearize the model

* Constraints and objective function can be represented as equality and inequality equations.

\begin{equation}
    x_{ns}= 
        \begin{cases}
            0, &\text{nurse n will not work on shift s}\ \\
            1, &\text{nurse n will work on shift s}\
        \end{cases}
\end{equation}

* s: each shift

* n: each nurse

* r: list storing the required nurses in each day.


\begin{equation}
    \text{r} = [\text{number of nurses required in specific shift s}] \\
    \text{r[even index]} = [\text{number of nurses required in day shifts}] \\
    \text{r[odd index]} = [\text{number of nurses required in night shifts}] 
\end{equation}

* PTO = Dictionary which stores the information of the off-work shifts(nurses requested PTO or other reasons)

\begin{equation}
    \text{PTO} = \{n : [\text{list of off-work shifts for nurse n}]\}
\end{equation}


* daily_shift: [0, 1, 2, ...] in this case. The last element is late night shift.

* working_shift = [1,..., max_nurse_shifts]: nurse works either 8, 16, 24, 32, 40 hours per week.

* planning_length: in this notebook, planning_length is flexible. 

* Objective: minimize the numbers of scheduled nurses after satisfying all constraints. 


* Implementing linear programming with pulp python package to find the solution of this constrained optimization problem. 


## Result:

![Alt text](/figs/5shifts4weeks/nurse_scheduling.png?raw=true "Optional Title")

## Nurse_schedule_2shifts_1week.ipynb
nurse scheduling code in jupyter notebook for the simple case

## nurse_scheduling_general.py
A more general case in python file

## figs
contain output figs of the scheduling

## utils
1. off_shift dictionary that can be updated
2. output the scheduling into csv