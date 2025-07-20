import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from utils import *
'''
Problem Statement

Suppose you are the product manager of the factory and you have the test results for some microchips on two different tests. 
- From these two tests, you would like to determine whether the microchips should be accepted or rejected. 
- To help you make the decision, you have a dataset of test results on past microchips, from which you can build a logistic regression model.
'''

# Load Data
x_train, y_train = load_data('data/ex2data2.txt')
plot_data(x_train, y_train, pos_label="Accepted", neg_label="Rejected")
plt.ylabel("Microchip Test 2")
plt.xlabel("Microchip Test 1")
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
#Compute Cost
def compute_cost(x,y,w,b,lambda_):
    m = x.shape[0]
    n = x.shape[1]
    cost = 0
    for i in range(m):
        z = np.dot(x[i], w) + b
        f = sigmoid(z)
        Loss = -y[i] * np.log(f) - (1 - y[i]) * np.log(1 - f)
        cost += Loss
    cost *= (1/m)
    for j in range(n):
        cost += (lambda_ / (2 * m)) * (w[j] **2)
    return cost

#Compute Gradient
def compute_gradient(x,y,w,b,lambda_):
    m,n = x.shape
    dj_dw = np.zeros(n)
    dj_db =0.
    for i in range(m):
        z = np.dot(x[i], w) + b
        f = sigmoid(z)
        dj_db += (f - y[i])
        for j in range(n):
            dj_dw[j] += (f - y[i]) * x[i][j]
    dj_dw *= (1/m)
    dj_db *= (1/m)
    for j in range(n):
        dj_dw[j] += (lambda_ / m) * w[j]
    return dj_db, dj_dw
# Gradient Descent
def gradient_descent(x,y,w_in,b_in,lambda_,alpha,iters):
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []
    for i in range(iters):
        dj_db, dj_dw = compute_gradient(x,y,w,b,lambda_)
        w += - alpha * dj_dw
        b += - alpha * dj_db
        J_history.append(compute_cost(x,y,w,b,lambda_))
        if(i % 1000 == 0):
            print(f"Cost after iteration {i}: {compute_cost(x,y,w,b,lambda_)}")
    return w,b,J_history
'''
   One way to fit the data better is to create more features from each data point. In the provided function `map_feature`, we will map the features into all polynomial terms of $x_1$ and $x_2$ up to the sixth power.
'''
def map_feature(x1, x2):
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append(x1**(i-j) * (x2**j))
    return np.stack(out, axis=1)
x = map_feature(x_train[:,0], x_train[:,1])
y = y_train
np.random.seed(1)
w = np.random.rand(x.shape[1])-0.5
b = 1.
lambda_ = 0.01
alpha = 0.01
iters = 10000 
w,b,J_history = gradient_descent(x,y,w,b,lambda_,alpha,iters)
print(f'Cost after training: {compute_cost(x,y,w,b,lambda_)}')
#Plot the Leanring Curve
plt.plot(J_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()


#Plot the decision boundary
plot_decision_boundary(w,b,x,y)

plt.title(f"Decision Boundary for Lambda = 0.01")
plt.xlabel("Microchip Test 1")
plt.ylabel("Microchip Test 2")
plt.show()

def predict(x,w,b):
    m,n = x.shape
    p = np.zeros(m)
    for i in range(m):
        z = np.dot(x[i], w) + b
        f = sigmoid(z)
        p[i] = f >= 0.5
    return p

p = predict(x, w, b)

print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))