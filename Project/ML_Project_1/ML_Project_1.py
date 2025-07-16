"""
Problem:
Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet.

You would like to expand your business to cities that may give your restaurant higher profits.
The chain already has restaurants in various cities and you have data for profits and populations from the cities.
You also have data on cities that are candidates for a new restaurant.
For these cities, you have the city population.
Can you use the data to help you identify which cities may potentially give your business higher profits?
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

#Load the Dataset:
X_train , Y_train = load_data()

#Visualize the Dataset:

plt.scatter(X_train,Y_train,marker = 'x', c = 'r')
plt.title("Visualize the Dataset")
plt.ylabel("Profit in $10,000")
plt.xlabel("Population of City in 10,000s")
plt.show()

#Compute the Cost Function:

def compute_cost(X, Y, w, b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * X[i] + b
        cost = cost + (f_wb - Y[i])**2
    cost = cost / (2 * m)
    return cost

#Compute the Gradient:

def compute_gradient(X, Y, w, b):
    m = X.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * X[i] + b
        dj_db = dj_db + (f_wb - Y[i])
        dj_dw = dj_dw + (f_wb - Y[i]) * X[i]
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw

#Gradient Descent:
def gradient_descent(x,y,w_in,b_in,alpha,iters):
    w = 0
    b = 0
    dj_db , dj_dw = compute_gradient(x,y,w_in,b_in)
    J_hist = []
    for i in range(iters):
        dj_db, dj_dw = compute_gradient(x,y,w,b)
        tmp_w = w - alpha * (dj_dw)
        tmp_b = b - alpha * (dj_db)
        w = tmp_w
        b = tmp_b
        if(i < 10000):
            J_hist.append(compute_cost(x,y,w,b))
        if(i % 100 == 0):
            print(f"The value of w,b at iterations No. {i} is w = {w}, b = {b} \n")
    return b,w,J_hist

# Run the code:
alpha = 0.01
w_start = 0.
b_start = 0.
iterations = 1500

b,w,J_hist = gradient_descent(X_train,Y_train,w_start,b_start,alpha,iterations)
#Plot the Learning Curve:
plt.plot(J_hist)
plt.xlabel("Iteration")
plt.ylabel("Cost J")
plt.title("Learning Curve")
plt.show()

#Plot the Result:
plt.scatter(X_train,Y_train,marker = 'x', c = 'r')
plt.plot(X_train,w * X_train + b, c = 'b')
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000")
plt.show()


