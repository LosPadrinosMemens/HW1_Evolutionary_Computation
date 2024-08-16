import random
import numpy as np
import sympy as sp

symbols = sp.symbols('x1 x2')
x1, x2 = sp.symbols('x1 x2')
f_A = -2*x1**2 + 3*x1*x2 - 1.5*x2**2 - 1.3
f_B = (4 - 2.1*x1**2 + x1**4/3)*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2
# f_C Rastrigin function, I do not know if 2D or 3D

def obj_function(x):
    """
    Example of a objective function to test algorithms
    """
    fx = x ** 2 - 40 * x + 50
    return fx

def eval_sympy(obj_func, x):
    """
    Parameters:
    obj_func (sympy exp or list): Objective function or list of derivates
    x (np array):  Array of values to substitute into the objective function

    Returns:
    - f(x) (float): The evaluated function value
    """
    if isinstance(obj_func, list):
        return np.array([eval_sympy(func, x) for func in obj_func])
        
    elif isinstance(f_A, sp.Expr):
        sorted_symbols = sorted(obj_func.free_symbols, key=lambda s: s.name)
        n_x = len(sorted_symbols)
        if len(x) != n_x:
            raise ValueError(f"Incompatible dimensions: Expected {n_x} values, but got {len(x)}.")
        subs_dict = {symbol: value for symbol, value in zip(sorted_symbols, x)}
        result = obj_func.subs(subs_dict)
    return float(result)

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
    Continuous Hill Climber Multiple Objective Minimization algorithm

    Parameters:
    - obj_func (sympy exp): objective function
    - delta (float): open ball radius
    - n_iter (int): number of iterations
    - x_init (float): optional, for initializating the search

    Returns (as a tuple):
    - x_best: x solution
    - fx_best: f(x)
    """
    n_x = len(obj_func.free_symbols)
    x_best = x_init if x_init is not None else np.random.randint(1, 101, size=n_x) # Step 1
    fx_best = eval_sympy(obj_func,x_best)
    i = 0                                                             # Step 2
    while i < n_iter:                                                 # Step 3
        x = gen_neighbor(x_best, delta)                               # Step 4
        fx = eval_sympy(obj_func,x)                          
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
    - p(np array): search direction (from gradient descent)
    - obj_func (sympy exp): objective function
    - nabla_obj_func (sympy exp): derivative from obj_func

    Returns:
    - final_t: optimal step size
    """
    flag = False                                                        # Step 3
    while flag is False:                                                # Step 4
        x_tp = x+t*p
        left_expr1 = eval_sympy(obj_func, x_tp)
        right_expr1 = eval_sympy(obj_func,x) + c1*t*eval_sympy(nabla_obj_func, x).T*p
        left_expr2 = eval_sympy(nabla_obj_func, x_tp).T*p
        right_expr2 = c2*eval_sympy(nabla_obj_func, x).T*p
        if left_expr1 > right_expr1:                                    # Step 5
            t = t/2                                                     # Step 6
        elif left_expr2 < right_expr2:                                  # Step 7
            t = t * 2                                                   # Step 8
        else:                                                           # Step 9
            flag = True                                                 # Step 10
    return t                                                            # Step 13

def grad_descent(obj_func, x_init, tol):
    """
    Gradient descent algorithm 

    Parameters:
    - obj_func (sympy exp): objective function
    - x_init (np.ndarray): Initialized values
    - tol ()

    Returns:
    - x_best: x solution
    """
    c = 0                                                                                    # Step 2
    grad_norm = tol + 1                  
    x_best = x_init
    sorted_symbols = sorted(obj_func.free_symbols, key=lambda s: s.name)
    n_x = len(sorted_symbols)
    nabla_obj_func = list()
    for i in range(n_x):
        nabla_obj_func.append(sp.diff(obj_func, sorted_symbols[i]))

    while grad_norm > tol:                                                                   # Step 3
        p = eval_sympy(nabla_obj_func, x_best)                                                           # Step 4
        t = wolfe_cond(t=1, x=x_best, obj_func=obj_func, nabla_obj_func=nabla_obj_func, p=p) # Step 5
        x_best = x_best + t * p                                                              # Step 6
        grad_norm = np.linalg.norm(p)                                                        # Step 7
    return x_best
