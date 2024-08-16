import random
import numpy as np

def obj_function(x):
    """
    Example of a objective function to test algorithms
    """
    fx = x ** 2 - 40 * x + 50
    return fx

# Point A
def gen_neighbor(x_best, delta):
    """
    Generates a random neighbor to x_best, works for single and multi objective minimization problems
    """
    if type(x_best) == int:
        size = 1
    else:
        size = x_best.shape
    return x_best + np.random.uniform(-delta, delta, size)

def hill_climber(obj_func, delta, n_iter, x_init = None):
    """
    Continuous Hill Climber Single Objective Minimization algorithm

    Parameters:
    - obj_func (func): function that takes an x an returns fx
    - delta (float): open ball radius
    - n_iter (int): number of iterations
    - x_init (float): optional, for initializating the search

    Returns (as a tuple):
    - x_best: x solution
    - fx_best: f(x)
    """
    x_best = x_init if x_init is not None else random.randint(1, 100) # Step 1
    fx_best = obj_func(x_best)
    i = 0                                                             # Step 2
    while i < n_iter:                                                 # Step 3
        x = gen_neighbor(x_best, delta)                               # Step 4
        fx = obj_func(x)                          
        if fx < fx_best:                                              # Step 5
            x_best, fx_best = x, fx                                   # Step 6
            i += 1                                                    # Step 7
    return x_best, fx_best

# Point B
def wolfe_cond(t, x, obj_func, nabla_obj_func, p, c1=0.0001, c2=0.9):
    """
    Wolfe condition algorithm to determine step size

    Parameters:
    - t(float): initial step size
    - x(float): current solution
    - p(np.ndarray): search direction (from gradient descent)
    - obj_func (func): function that takes an x an returns fx
    - nabla_obj_func (np.ndarray): derivative from obj_func

    Returns:
    - final_t: optimal step size
    """
    flag = False                                                        # Step 3
    while flag is False:                                                # Step 4
        if obj_func(x+t*p) > obj_func(x) + c1*t*nabla_obj_func(x).T*p:  # Step 5
            t = t/2                                                     # Step 6
        elif nabla_obj_func(x+t*p).T*p < c2*nabla_obj_func(x).T*p:      # Step 7
            t = t * 2                                                   # Step 8
        else:                                                           # Step 9
            flag = True                                                 # Step 10
    return t                                                            # Step 13

def grad_descent(obj_func, nabla_obj_func, x_init, tol):
    """
    Gradient descent algorithm 

    Parameters:
    - obj_func (func): function that takes an x an returns fx
    - nabla_obj_func (np.ndarray): derivative from obj_func
    - x_init (np.ndarray): Initialized values
    - tol ()

    Returns:
    - x_best: x solution
    """
    c = 0                                                                                    # Step 2
    grad_norm = tol + 1                  
    x_best = x_init                  
    while grad_norm > tol:                                                                   # Step 3
        p = nabla_obj_func(x_best)                                                           # Step 4
        t = wolfe_cond(t=1, x=x_best, obj_func=obj_func, nabla_obj_func=nabla_obj_func, p=p) # Step 5
        x_best = x_best + t * p                                                              # Step 6
        grad_norm = np.linalg.norm(p)                                                        # Step 7
    return x_best
