from __future__ import division, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

# Height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# Weight (kg)
y = np.array([[49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# Buildding Xbar
one = np.ones((X.shape[0], 1)) # Return one matrix [nx1] consist all 1
Xbar = np.concatenate((one, X), axis = 1) # Concatenate one and X => Xbar

# Calculating weights of the fitting Line
A = np.dot(Xbar.T, Xbar) # Multi Matrix
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b) # Pseudo inverse - Gia nghich dao
#print 'w = ', w

# Preparing the fitting line
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1 * x0

# Visualize data
plt.plot(X, y, 'ro')
plt.plot(x0, y0)
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept = False)
regr.fit(Xbar, y)

# Compare two Results :
print 'Solution found by scikit-learn : ', regr.coef_
print 'Solution found by (5) : ', w.T
