import random
import numpy as np
import sympy as sp

symbols = sp.symbols('x1 x2')
x1, x2 = sp.symbols('x1 x2')
f_A_orig = -2*x1**2 + 3*x1*x2 - 1.5*x2**2 - 1.3
f_A = -f_A_orig
f_B = (4 - 2.1*x1**2 + x1**4/3)*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2
f_C = 20 + (x1**2 - 10*sp.cos(2*sp.pi*x1)) + (x2**2 - 10*sp.cos(2*sp.pi*x2)) # Rastrigin function

#####################
### AUX FUNCTIONS ###
#####################
def eval_sympy(obj_func, x):
    """
    Parameters:
    obj_func (sympy exp or list): Objective function or list of derivates
    x (np.ndarray):  Array of values to substitute into the objective function

    Returns:
    - f(x) (float): The evaluated function value
    """
    if isinstance(obj_func, list):
        if all(isinstance(el, list) for el in obj_func): # Hessian Matrix
            return np.array([[eval_sympy(func, x) for func in row] for row in obj_func])
        else:                                            # Jacobian Vector
            return np.array([eval_sympy(func, x) for func in obj_func])
    
    elif isinstance(obj_func, sp.Expr):
        sorted_symbols = sorted(obj_func.free_symbols, key=lambda s: s.name)
        n_x = len(sorted_symbols)
        subs_dict = {symbol: value for symbol, value in zip(sorted_symbols, x)}
        result = obj_func.subs(subs_dict)
    return float(result)

def constraint_checker(x, constraints):
    """
    Checks if the new the solution of the problem is within the constraint of a problem

    Parameters:
    - x (np.ndarray): New solution to the problem
    - constraints (list of list): Defining lower and upper limits for each variable e.g. [[-3, 3], [-2, 2]]

    Returns:
    - Boolean: True if solution is within the constraints, False otherwise
    """
    if len(constraints) == 0:
        return True  # No constraints provided, always return True
    
    for i in range(len(x)):
        if not (constraints[i][0] <= x[i] <= constraints[i][1]):
            return False
    return True

def print_verbose(verbose_level, x_best, i):
    """
    For printing progress in search algorithms
    """
    if verbose_level == 1:
        print(f'i = {i}, ' + ', '.join([f'x{idx+1} = {val}' for idx, val in enumerate(x_best)]), end='\r')
    elif verbose_level == 2:
        print(f'i = {i}, ' + ', '.join([f'x{idx+1} = {val}' for idx, val in enumerate(x_best)]))

####################
### HILL CLIMBER ###
####################
def gen_neighbor(x_best, delta):
    """
    Generates a random neighbor to x_best, works for single and multi objective minimization problems
    """
    if type(x_best) == int:
        size = 1
    else:
        size = x_best.shape
    return x_best + np.random.uniform(-delta, delta, size)

def hill_climber(obj_func, delta, n_iter, x_init = None, constraints = None, n_intervals = 15, verbose = 1):
    """
    Continuous Hill Climber Multiple Objective Minimization algorithm

    Parameters:
    - obj_func (sympy exp): objective function
    - delta (float): open ball radius
    - n_iter (int): number of iterations
    - x_init (np.ndarray): optional, for initializating the search
    - constraints (list of list): Defining lower and upper limits for each variable
    - verbose (int): verbose levels 0, 1 and 2 from less verbose to more verbose
    - n_intervals (int): number of tracking points

    Returns (as a tuple):
    - x_best: x solution
    - fx_best: f(x)
    - x_history: list of numpy arrays with solutions
    - fx_history: list of numpy arrays with values of f_x_best
    """
    n_x = len(obj_func.free_symbols)
    x_best = x_init if x_init is not None else np.array([np.random.uniform(low, high) for low, high in constraints]) # Step 1
    fx_best = eval_sympy(obj_func,x_best)
    i = 0                                                             # Step 2
    x_history = []
    fx_history = []

    while i < n_iter:                                                 # Step 3
        x = gen_neighbor(x_best, delta)                               # Step 4
        fx = eval_sympy(obj_func,x)                          
        if (constraints is None or constraint_checker(x, constraints)) and (fx < fx_best): # Step 5
            x_best, fx_best = x, fx                                   # Step 6
        i += 1                                                        # Step 7

        if (i % (n_iter // n_intervals) == 0) or (i == n_iter):
            x_history.append(x_best.copy())
            fx_history.append(fx_best)
            if verbose > 0 : # Print n_intervals times and in the last iter
                print_verbose(verbose, x_best, i)

    return x_best, fx_best, x_history, fx_history

########################
### GRADIENT DESCENT ###
########################
def wolfe_cond(t, x, obj_func, nabla_obj_func, p, c1=0.0001, c2=0.9, max_wolf_iter = 8):
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
    i = 0 
    flag = False                                                        # Step 3
    
    #We can compute it here to reduce the computation.
    grad_x = eval_sympy(nabla_obj_func, x)
    grad_x_dot_p = np.dot(grad_x,p)
    f_x = eval_sympy(obj_func,x)
    right_expr2 = c2*grad_x_dot_p
    
    while (flag is False) and (i < max_wolf_iter):                                                # Step 4
        #xtp = current_x + step*dir
        x_tp = x+t*p 
        # f(xtp) <= f(x) + c1*t*grad*p
        left_expr1 = eval_sympy(obj_func, x_tp) 
        
        # grad_xtp'*p >= c2*grad_x*p
        right_expr1 = f_x + c1*t*grad_x_dot_p
        left_expr2 = np.dot(eval_sympy(nabla_obj_func, x_tp),p)
        print(f't = {t}', end='\r')

        # Comparisons seem reversed, but it is correct LoL
        if left_expr1 >= right_expr1:                                    # Step 5
            t = t / 2                                                     # Step 6
        elif left_expr2 <= right_expr2:                                  # Step 7
            t = t * 2                                                   # Step 8
        else:                                                           # Step 9
            flag = True                                                 # Step 10

        #Included this max step so that it wont go to 0 or infinite.
        if t < 0.0001:
            return 0.0001
        if t > 2.01:
            return 2
        i += 1
    #print(f't = {t}')
    return t                       

def grad_descent(obj_func, tol, x_init = None, max_iter = 1000, constraints = None, n_intervals = 15, verbose = 1):
    """
    Gradient descent algorithm 

    Parameters:
    - obj_func (sympy exp): objective function
    - x_init (np.ndarray): Initialized values
    - tol (float): tolerance for the grad_norm

    Returns:
    - x_best: x solution
    - fx_best: f(x)
    - x_history: list of numpy arrays with solutions
    - fx_history: list of numpy arrays with values of f_x_best

    """
    grad_norm = tol + 1                  
    sorted_symbols = sorted(obj_func.free_symbols, key=lambda s: s.name)
    n_x = len(sorted_symbols)
    x_best = x_init if x_init is not None else np.array([np.random.uniform(low, high) for low, high in constraints]) # Step 1
    fx_best = eval_sympy(obj_func,x_best)
    x_history = []
    fx_history = []

    nabla_obj_func = list()
    for i in range(n_x):
        nabla_obj_func.append(sp.diff(obj_func, sorted_symbols[i]))
        
    i = 0                                                                                     # Step 2
    while grad_norm > tol and i < max_iter:                                                   # Step 3
        p = -eval_sympy(nabla_obj_func, x_best)                                               # Step 4
        t = wolfe_cond(t=1, x=x_best, obj_func=obj_func, nabla_obj_func=nabla_obj_func, p=p)  # Step 5
        x = x_best + t * p                                                                    # Step 6
        grad_norm = np.linalg.norm(p)                                                         # Step 7
        fx = eval_sympy(obj_func,x)
        
        if (constraints is None or constraint_checker(x, constraints)) and (fx < fx_best):        # Step 5
            x_best, fx_best = x, fx
        i += 1

        if (i % (max_iter // max_iter) == 0) or (i == max_iter) or (grad_norm > tol):
            x_history.append(x_best.copy())
            fx_history.append(fx_best)
            if verbose > 0 : # Print n_intervals times and in the last iter
                print_verbose(verbose, x_best, i)
        
    return x_best, fx_best, x_history, fx_history

#####################
### NEWTON METHOD ###
#####################
def newton_method(obj_func, tol, x_init = None, max_iter = 1000, constraints = None, n_intervals = 15, verbose = 1):
    """
    Gradient descent algorithm 

    Parameters:
    - obj_func (sympy exp): objective function
    - x_init (np.ndarray): Initialized values
    - tol (float): tolerance for the grad_norm

    Returns:
    - x_best: x solution
    - fx_best: f(x)
    - x_history: list of numpy arrays with solutions
    - fx_history: list of numpy arrays with values of f_x_best

    """
    grad_norm = tol + 1                  
    sorted_symbols = sorted(obj_func.free_symbols, key=lambda s: s.name)
    n_x = len(sorted_symbols)
    x_best = x_init if x_init is not None else np.array([np.random.uniform(low, high) for low, high in constraints]) # Step 1
    fx_best = eval_sympy(obj_func,x_best)
    x_history = []
    fx_history = []

    nabla_obj_func = list()
    for i in range(n_x):
        nabla_obj_func.append(sp.diff(obj_func, sorted_symbols[i]))
    
    hessian_obj_func = []
    for i in range(n_x):
        hessian_row = []
        for j in range(n_x):
        # Compute the second derivative with respect to variables i and j
            second_derivative = sp.diff(obj_func, sorted_symbols[i], sorted_symbols[j])
            hessian_row.append(second_derivative)
        hessian_obj_func.append(hessian_row)
    
    i = 0                                                                                     # Step 2
    while grad_norm > tol and i < max_iter:                                                   # Step 3
        hessian_eval_inv = np.linalg.inv(eval_sympy(hessian_obj_func, x_best))
        p = -np.dot(hessian_eval_inv, eval_sympy(nabla_obj_func, x_best))                     # Step 4
        #t = wolfe_cond(t=1, x=x_best, obj_func=obj_func, nabla_obj_func=nabla_obj_func, p=p) # Step 5
        x = x_best + p                                                                        # Step 6
        grad_norm = np.linalg.norm(p)                                                         # Step 7
        fx = eval_sympy(obj_func,x)
        
        if (constraints is None or constraint_checker(x, constraints)) and (fx < fx_best):        # Step 5
            x_best, fx_best = x, fx
        i += 1

        if (i % (max_iter // max_iter) == 0) or (i == max_iter) or (grad_norm > tol):
            x_history.append(x_best.copy())
            fx_history.append(fx_best)
            if verbose > 0 : # Print n_intervals times and in the last iter
                print_verbose(verbose, x_best, i)
        
    return x_best, fx_best, x_history, fx_history

#######################
### VALIDATION PART ###
#######################
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
from scipy.optimize import differential_evolution

def plot_function_with_paths(obj_func, constraints, x_history, fx_history):
    """
    Plot a 2D contour plot of the given function with both the true minimum found by differential evolution
    and the precomputed optimization path from another method.

    Parameters:
    - obj_func (sympy exp): Objective function to minimize.
    - constraints (list of lists): Range constraints [[x1_min, x1_max], [x2_min, x2_max]].
    - x_history (list of np.ndarray): List of coordinates representing the path of the optimization from another method.
    - fx_history (list of float): List of function values corresponding to each point in x_history.

    Returns:
    - None
    """
    x1_min, x1_max = constraints[0]
    x2_min, x2_max = constraints[1]
    
    # Create a grid of points
    x1_vals = np.linspace(x1_min, x1_max, 100)
    x2_vals = np.linspace(x2_min, x2_max, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Convert the sympy function to a lambda function for evaluation
    f_A_lambdified = sp.lambdify((sp.symbols('x1'), sp.symbols('x2')), obj_func, 'numpy')

    # Evaluate the function on the grid
    Z = f_A_lambdified(X1, X2)

    # Perform global optimization using differential evolution to find the true minimum
    def obj_func_numpy(x):
        return f_A_lambdified(x[0], x[1])

    # Define bounds for the differential evolution
    bounds = [(x1_min, x1_max), (x2_min, x2_max)]

    # Run the differential evolution algorithm
    result = differential_evolution(obj_func_numpy, bounds)

    # Convert x_history to a numpy array for easier indexing
    x_history = np.array(x_history)

    # Plotting the 2D contour plot
    plt.figure(figsize=(8, 6))
    plt.contour(X1, X2, Z, levels=30, cmap='viridis')
    
    # Plot the precomputed path from your model
    plt.plot(x_history[:, 0], x_history[:, 1], 'ro-', linewidth=2, markersize=4, label='Model Path')
    
    # Mark the true minimum found by differential evolution
    plt.scatter(result.x[0], result.x[1], color='blue', s=100, label='True Minimum (Differential Evolution)')
    
    # Mark the minimum found by your model
    plt.scatter(x_history[-1, 0], x_history[-1, 1], color='red', s=100, label='Model Minimum')

    plt.xlim([x1_min, x1_max])
    plt.ylim([x2_min, x2_max])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour Plot with Optimization Paths')
    
    plt.legend()
    plt.show()
    
    print(f"True minimum found by differential evolution at x1 = {result.x[0]:.2f}, x2 = {result.x[1]:.2f}, with value = {result.fun:.2f}")
    print(f"Model minimum at x1 = {x_history[-1, 0]:.2f}, x2 = {x_history[-1, 1]:.2f}, with value = {fx_history[-1]:.2f}")