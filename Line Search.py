
"""Volume II Lab 15: Line Search Algorithms
Allyson Tom
"""

"""
Investigate various Line-Search algorithms for numerical optimization.
"""

import numpy as np
from scipy import linalg as la
from scipy.optimize import leastsq
from matplotlib import pyplot as plt


def newton1d(f, df, ddf, x, niter=10):
    """
    Perform Newton's method to minimize a function from R to R.

    Parameters:
        f (function): The twice-differentiable objective function.
        df (function): The first derivative of 'f'.
        ddf (function): The second derivative of 'f'.
        x (float): The initial guess.
        niter (int): The number of iterations. Defaults to 10.
    
    Returns:
        (float) The approximated minimizer.
    """

    x_old = float(x)
    i = 1
    while i <= niter:
        x_new = x_old - (float(df(x_old))/ddf(x_old))
        x_old = x_new
        i+=1
    return x_new
    


def test_newton():
    """Use the newton1d() function to minimixe f(x) = x^2 + sin(5x) with an
    initial guess of x_0 = 0. Also try other guesses farther away from the
    true minimizer, and note when the method fails to obtain the correct
    answer.

    Returns:
        (float) The true minimizer with an initial guess x_0 = 0.
        (float) The result of newton1d() with a bad initial guess.
    """
    function = lambda x: x**2+np.sin(5*x)
    dfunction = lambda x: 2*x+5*np.cos(5*x)
    ddfunction = lambda x: 2 - 25*np.sin(5*x)
    x_guess1 = 0.
    x_guess2 = 10.
    return newton1d(function, dfunction, ddfunction, x_guess1), newton1d(function, dfunction, ddfunction, x_guess2)



def backtracking(f, slope, x, p, a=1, rho=.9, c=10e-4):
    """Perform a backtracking line search to satisfy the Armijo Conditions.

    Parameters:
        f (function): the twice-differentiable objective function.
        slope (float): The value of grad(f)^T p.
        x (ndarray of shape (n,)): The current iterate.
        p (ndarray of shape (n,)): The current search direction.
        a (float): The intial step length. (set to 1 in Newton and
            quasi-Newton methods)
        rho (float): A number in (0,1).
        c (float): A number in (0,1).
    
    Returns:
        (float) The computed step size satisfying the Armijo condition.
    """
    while (f(x+a*p)) > (f(x)+c*a*slope):
        a = float(rho*a)
    return a


  
def gradientDescent(f, df, x, niter=10):
    """Minimize a function using gradient descent.

    Parameters:
        f (function): The twice-differentiable objective function.
        df (function): The gradient of the function.
        x (ndarray of shape (n,)): The initial point.
        niter (int): The number of iterations to run.
    
    Returns:
        (list of ndarrays) The sequence of points generated.
    """
    
    points = [x]
    i = 1
    while i <= niter:
        slope = np.dot(df(x), -df(x))
        pk = -df(x)
        a = backtracking(f, slope, x, pk)
        xnew = x + a*pk
        x = xnew
        points.append(x)
        i+=1
    return points


def newtonsMethod(f, df, ddf, x, niter=10):
    """Minimize a function using Newton's method.

    Parameters:
        f (function): The twice-differentiable objective function.
        df (function): The gradient of the function.
        ddf (function): The Hessian of the function.
        x (ndarray of shape (n,)): The initial point.
        niter (int): The number of iterations.
    
    Returns:
        (list of ndarrays) The sequence of points generated.
    """
    points = [x]
    i = 1
    while i <= niter:
        pk = -np.dot(la.inv(ddf(x)),df(x))
        slope = np.dot(df(x), pk)
        a = backtracking(f, slope, x, pk)
        xnew = x + a*pk
        x = xnew
        points.append(x)
        i+=1
    return points



def gaussNewton(f, df, jac, r, x, niter=10):
    """Solve a nonlinear least squares problem with Gauss-Newton method.

    Parameters:
        f (function): The objective function.
        df (function): The gradient of f.
        jac (function): The jacobian of the residual vector.
        r (function): The residual vector.
        x (ndarray of shape (n,)): The initial point.
        niter (int): The number of iterations.
    
    Returns:
        (ndarray of shape (n,)) The minimizer.
    """

    k = 1
    while k <= niter:
        p = la.solve(np.dot(jac(x).T,jac(x)),-np.dot(jac(x).T,r(x)))
        a = optimize.line_search(f, df, x, p)[0]
        newx = x + a*p
        x = xnew
        k+=1
    return xnew



def census():
    """Generate two plots: one that considers the first 8 decades of the US
    Census data (with the exponential model), and one that considers all 16
    decades of data (with the logistic model).
    """

    # Start with the first 8 decades of data.
    years1 = np.arange(8)
    pop1 = np.array([3.929,  5.308,  7.240,  9.638,
                    12.866, 17.069, 23.192, 31.443])

    # Now consider the first 16 decades.
    years2 = np.arange(16)
    pop2 = np.array([3.929,   5.308,   7.240,   9.638,
                    12.866,  17.069,  23.192,  31.443,
                    38.558,  50.156,  62.948,  75.996,
                    91.972, 105.711, 122.775, 131.669])


    def model (x, t):
        return x[0]*np.exp(x[1]*(t+x[2]))
    def residual(x):
        return model(x,years1) - pop1
    x0 = np.array([150.,0.4,2.5])
    x = leastsq(residual,x0)[0]

    #plot
    dom = np.linspace(0,8,600)
    y = model(x,dom)
    plt.plot(years1, pop1, '*')
    plt.plot(dom, y, '--')
    plt.show()


    x02 = np.array([150.,.4,-15.])
    def model2(x,t):
        return float(x[0])/(1+np.exp(-x[1]*(t+x[2])))
    def residual2(x):
        return model2(x,years2) - pop2
    x2 = leastsq(residual2,x02)[0]

    #plot
    dom2 = np.linspace(0,16,600)
    y2 = model2(x2,dom2)
    plt.plot(years2,pop2, '*')
    plt.plot(dom2,y2, '--')
    plt.show()




