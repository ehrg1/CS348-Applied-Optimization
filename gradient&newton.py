import numpy as np

# Objective function: f(x) = x^2 + 4x + 4
def f(x):
    return x**2 + 4*x + 4

# Gradient: f'(x) = 2x + 4
def grad_f(x):
    return 2*x + 4

# Hessian: f''(x) = 2
def hess_f(x):
    return 2

# Gradient Descent
def gradient_descent(x0, lr=0.1, max_iter=50):
    x = x0
    path = [x]
    for _ in range(max_iter):
        x = x - lr * grad_f(x)
        path.append(x)
    return path



#Newton's Method
def newtons_method(x0, maxiter=10):
    x = x0
    path = [x]
    for _ in range(max_iter):
        x = x - grad_f(x) / hess_f(x)
        path.append(x)
    return path


# Initialize and run
x0 = 10
gd_path = gradient_descent(x0)
newton_path = newtons_method(x0)
