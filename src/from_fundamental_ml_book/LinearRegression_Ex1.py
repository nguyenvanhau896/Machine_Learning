from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
# height (cm), input data, each row is a data point
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])
""" (cân nặng) = w_1*(chiều cao) + w_0
ở đây w_0 chính là bias b.
"""
"""giả nghịch đảo của một ma trận A trong Python 
sẽ được tính bằng numpy.linalg.pinv(A).
"""
# Building XBar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1) # Each point is one row
# Calculating weights of the fitting line
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
# weights 
w_0, w_1 = w[0], w[1]

from sklearn import datasets, linear_model
# fit the model by Linear Regression
regr = linear_model.LinearRegression()
regr.fit(X, y) # in scikit-learn, each sample is one row
# Compare two results
print("scikit-learn’s solution : w_1 = ", regr.coef_[0], "w_0 = ", regr.intercept_)
print("our solution : w_1 = ", w[1], "w_0 = ", w[0])

# def w(h):
#     return w_1 * h + w_0
# h = np.linspace(140, 190, 1000)
# w = w(h)

# plt.scatter(X, y, c='red')
# plt.plot(h, w)
# plt.xlabel("Height (cm)")
# plt.ylabel("Weight (kg)")
# plt.show()
