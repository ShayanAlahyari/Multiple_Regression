# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Creating the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Handling categorical predictors
encoder = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x = np.array(encoder.fit_transform(x))

# Splitting the data into training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# Creating the model

regressor = LinearRegression()
regressor.fit(x_train,y_train)

# Plotting
y_predicted = regressor.predict(x_train)
np.set_printoptions(precision=2)
# print(np.concatenate((y_predicted.reshape(len(y_predicted),1),y_test.reshape(len(y_test),1)),0))

y_train_pred = regressor.predict(x_train)
y_test_pred = regressor.predict(x_test)
# Plotting residuals for training set
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_train_pred, train_residuals, color='blue', label='Training data')
plt.scatter(y_test_pred, test_residuals, color='red', label='Testing data')
plt.hlines(0, min(y_train_pred), max(y_train_pred), color='black', lw=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.legend()
plt.show()



# Plotting Actual vs. Predicted values for training set
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, color='blue', label='Training data')
plt.scatter(y_test, y_test_pred, color='red', label='Testing data')
plt.plot([min(y), max(y)], [min(y), max(y)], color='black', lw=2)  # Line for perfect prediction
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()