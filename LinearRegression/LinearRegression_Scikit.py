from sklearn import datasets, linear_model

# Height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# Weight (kg)
y = np.array([[49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept = False)
regr.fit(Xbar, y)

print regr.coef_