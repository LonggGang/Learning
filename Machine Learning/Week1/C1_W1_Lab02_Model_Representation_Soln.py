import numpy as np
import matplotlib.pyplot as plt
# Use a default matplotlib style to avoid missing file errors
plt.style.use('ggplot')
x_train = np.array([1.0, 2.0])   # features
y_train = np.array([300.0, 500.0]) # target value
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

plt.scatter(x_train ,y_train ,marker='x',c ='r')
#Set the title
plt.title("Housing Prices")
plt.ylabel("Price (in 1000s of dollars)")
plt.xlabel("Size (in 1000 sqft)")
plt.show()
w = 200
b = 100
# For this train data, w and b have to be 200 and 100
def compute_model_output(x,w,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb
tmp_f_wb = compute_model_output(x_train,w,b)
plt.plot(x_train,tmp_f_wb,c='b',label = 'Our Prediction')
plt.scatter(x_train,y_train,marker='x',c='r')
plt.title("Housing Prices")
plt.ylabel("Price (in 1000s of dollars)")
plt.xlabel("Size (in 1000 sqft)")
plt.legend()
plt.show()
x_i = 1.2
cost_1200sqft = w * x_i + b    
print(f"${cost_1200sqft:.0f} thousand dollars")
