# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Dataset of Daily coronavirus infections from 7 dec 2020 to 5 jan 2021 in south africa
# x = Days
# y = Number of recorded infections 
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

y = [3313,4014,6709,8166,8319,7882,7999,5163,7552,10008,9126,8725,10939,9445,8789,9501,14046,14305,14796,11552,9502,7458,9580,17710,18000,16726,15002,11859,12601,14410]

# Split data into training and testing sets using skikit-Learn train_test_split function
# we also specify how much data we want tested and how we shuffle the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)



# Convert data to array 
x_train = np.array(x_train)
y_train = np.array(y_train)

# Increase array dimention to 2D 
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

# Reorder the array
y_train = y_train[x_train[:,0].argsort()]
x_train = x_train[x_train[:, 0].argsort()]

# Set degree of the polynomial to 3
poly = PolynomialFeatures(degree = 3)

# Transform input data matrix to new matrix of given degree 
x_poly = poly.fit_transform(x_train)

# Train model 
poly_reg = LinearRegression()
poly_reg.fit(x_poly, y_train)

# Plot data together with line
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
plt.scatter(x_test, y_test, c = '#edbf6f', label = 'Testing Data')
plt.scatter(x_train, y_train, c = '#8acfd4', label = 'Training Data')
plt.legend(loc="upper left")
plt.title('30 ')
plt.title('Daily Corona Virus Cases in South Africa(7 December - 5 January)')
plt.xlabel('Days')
plt.ylabel('Cases')
plt.plot(x_train, poly_reg.predict(x_poly), c='#a3cfa3', label='Polynomial regression line')
plt.legend(loc="upper left")
plt.show()