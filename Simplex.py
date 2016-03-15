# name this file 'solutions.py'.
"""Volume II Lab 16: Simplex
Allyson Tom


Implements the Simplex Algorithm to solve linear constrained optimization problems.
"""

import numpy as np


class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        maximize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    
    def __init__(self, c, A, b):
        """
        Parameters:
            c (1xn ndarray): The coefficients of the linear objective function.
            A (mxn ndarray): The constraint coefficients matrix.
            b (1xm ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        self.c = c
        self.A = A
        self.b = b
        self.L = range(A.shape[1],A.shape[1] + A.shape[0]) + range(0,A.shape[1])
        self.m = len(A)
        self.n = len(A.T)

        #check for feasibility
        check = np.all(np.less_equal(np.zeros(len(b)),b))
        if check == False:
                raise ValueError("Problem is not feasible at the origin.")
        


        #Create the tableau
        """Abar = np.hstack([A,np.eye(self.m)])
        cbar = np.hstack([-c,np.zeros(self.m)]).T
        matrix = np.vstack([cbar, Abar])
        matrix = np.column_stack((b,Abar))
        matrix = np.column_stack((matrix, np.zeros(self.m)))
        top = np.hstack([0,c])
        top = np.hstack([top,self.m])
        top = np.hstack([top,1])
        self.tableau = np.vstack([top,matrix])
        self.tableau = np.array(self.tableau)"""
        c = np.hstack((0,self.b))
        mtop = np.hstack((-self.c.T,np.zeros(self.m)))
        mbottom = np.hstack((self.A,np.eye(self.m)))
        mid = np.vstack([mtop,mbottom])
        beg = np.column_stack([c.T,mid])
        end = np.hstack((1,np.zeros(self.m)))
        self.tableau = np.column_stack([beg,end.T])

    def find_entering(self):
        """
        i = 0 #row
        j = 1 #column
        while i <= (self.m-1):
            while j <= (self.n-1):
                if self.tableau[i][j] < 0:
                    return j-1
                j+=1
            i+=1
        """
        column = np.argmax(self.tableau[0,:]<0)
        if np.all(self.tableau[:, column]<0):
            raise ValueError("Problem is unbounded")
        else:
            #calculates the ratios
            div = np.copy(self.tableau[1:, column])
            div[div <0] = 0
            ratios = self.tableau[1:,0]/div
            row = np.argmin(abs(ratios))+1
        return row, column
        
    def pivot(self):
        row, column = self.find_entering()
        basic = self.L.index(column - 1)
        nonbasic = self.L.index(row + 1)
        self.L[basic] = self.L[nonbasic]
        self.L[nonbasic] = self.L[basic]

        self.tableau[row,:] = self.tableau[row,:]/self.tableau[row][column]
        for k in xrange(row):
            self.tableau[k,:] = -self.tableau[k,column]*self.tableau[row,:] + self.tableau[k,:]
        for k in xrange(row+1, self.tableau.shape[0]):
            self.tableau[k,:] = -self.tableau[k,column]*self.tableau[row,:]+self.tableau[k,:]


    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The maximum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        """def find_entering(self):
            i = 0 #row
            j = 1 #column
            while i <= (self.m-1):
                while j <= (self.n-1):
                    if self.tableau[i][j] < 0:
                        return j-1
                    j+=1
                i+=1
        j = find_entering()
        if np.any(self.tableau[:,j]<=0):
            raise ValueError("Problem is unbounded.")
        ratios = []
        k = 0
        for entry in self.tableau[:,0]:
            ratios.append(self.tableau[k][0]/float(entry))
            k += 1"""

        while np.any(self.tableau[0,:]<0):
            self.pivot()

        maximum = self.tableau[0][0]
        basic = {}
        nonbasic = {}
        a, b = np.shape(self.tableau)
        for i in range(1, b-1):
            if self.tableau[0,i] != 0:
                nonbasic[i-1] = 0
            else:
                for j in range(a):
                    if self.tableau[j,i] == 1:
                        basic[i-1] = self.tableau[j,0]
        return maximum, basic, nonbasic

        """
        solution = self.tableau[0][0]
        optimizers = self.tableau[1:,0]
        basic_index = self.L[:len(self.b)]
        nonbasic_index = np.zeros(len(self.c))
        basic_var_dict = dict(zip(basic_index,optimizers))
        nonbasic_var_dict = dict(zip(self.L[len(self.b):],nonbasic_index))
        return solution, basic_var_dict, nonbasic_var_dict
        """


        



def prob7(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        The minimizer of the problem (as an array).
    """
    productmix = np.load("productMix.npz")
    A = productmix['A']
    p = productmix['p']
    m = productmix['m']
    d = productmix['d']

    A = np.vstack((A,np.eye(A.shape[1])))
    b = np.hstack((m,d))
    optimize = SimplexSolver(p,A,b)

    maximum, basic, nonbasic = optimize.solve()

    n = len(p)
    answer = np.zeros(n)
    for i in xrange(n):
        if basic.has_key(i):
            answer[i] = basic[i]
        else:
            answer[i] = nonbasic[i]
    return answer


    return 


# END OF FILE =================================================================


