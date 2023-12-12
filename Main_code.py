# -*- coding: utf-8 -*-
"""M22EE051_Question1.ipynb
"""

#upload file in google colab
from google.colab import files
uploaded = files.upload()

#importing recommended libraries
import pandas as pd
import numpy as np

# Load the student data from a CSV file into a DataFrame
df = pd.read_csv('student_data.csv')

# Converting categorical variable to numerical using one-hot encoding
df = pd.get_dummies(df, columns=["Extracurricular Activities"], drop_first=True)

"""# **Q1.1 Identify the dependent and independent variables.**"""

# Spliting the DataFrame into features (X) and target variable (y)

X = df.drop("Performance", axis=1)  # Features/independent variables
y = df["Performance"]  # Target variable/dependent variable

"""# **Q1.3 Split the data set in train and test (80:20) ratio.**"""

# Set a random seed for reproducibility in random operations
np.random.seed(2)

# Shuffle the indices of the data to ensure randomness
indices = np.arange(len(df))
np.random.shuffle(indices)

# Calculate the index to split the data between training and testing sets
split_idx = int(0.8 * len(df))  # 80% for training, 20% for testing

# Split the indices into training and testing indices
train_indices = indices[:split_idx]  # Indices for training set
test_indices = indices[split_idx:]    # Indices for testing set

# Define the column labels representing the features in the DataFrame
feature_columns = ['Hours Studied', 'Previous Scores', 'Duration of Sleep', 'Sample Question Papers Practiced', 'Extracurricular Activities_Yes']

# Extract training and testing data based on the shuffled indices and feature columns
X_train = (df.loc[train_indices, feature_columns]).values  # Training features
y_train = (df.loc[train_indices, 'Performance']).values   # Training target variable
X_test = (df.loc[test_indices, feature_columns]).values   # Testing features
y_test = (df.loc[test_indices, 'Performance']).values     # Testing target variable

X_train.shape

X_test.shape

"""# **Q1.4 Write a python code for Linear Regression (from scratch) and train the model with training data.**"""

#class of Linear Regression by Gradient Descent
class GradientDescent:

    def __init__(self, learning_rate=0.01, epochs=100):
        # Initialize the regression model with hyperparameters
        self.coef_ = None  # Coefficients for each feature
        self.intercept_ = None  # Intercept term
        self.lr = learning_rate  # Learning rate for gradient descent
        self.epochs = epochs  # Number of training epochs

    def fit(self, X_train, y_train):
        # Initialize coefficients and intercept
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])  # Initialize coefficients with ones
        self.losses = []  # To store loss at each epoch

        for i in range(self.epochs):
            epoch_loss = 0  # Loss for the current epoch
            # Calculate predicted target values (y_hat) using current coefficients
            y_hat = np.dot(X_train, self.coef_) + self.intercept_

            # Calculate derivative of the intercept for gradient update
            intercept_der = -1 * np.mean(y_train - y_hat)
            self.intercept_ = self.intercept_ - (self.lr * intercept_der)  # Update intercept using gradient descent

            # Calculate derivative of the coefficients for gradient update
            coef_der = -1 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]
            self.coef_ = self.coef_ - (self.lr * coef_der)  # Update coefficients using gradient descent

            # Calculate and accumulate the loss for the current sample
            epoch_loss += (np.mean((y_train - y_hat)**2))/2

            # Store the calculated loss for current epoch
            self.losses.append(epoch_loss)

        # Print the learned intercept and coefficients after training
        print("Intercept:", self.intercept_)
        print("Coefficients:", self.coef_)

    def predict(self, X_test):
        # Predict target values using learned coefficients and intercept
        return np.dot(X_test, self.coef_) + self.intercept_

#class of Linear Regression by Stochastic Gradient Descent
class Stoc_Grad_Desc:

    def __init__(self, learning_rate=0.01, epochs=100):
        # Initialize the regression model with hyperparameters
        self.coef_ = None  # Coefficients for each feature
        self.intercept_ = None  # Intercept term
        self.lr = learning_rate  # Learning rate for stochastic gradient descent
        self.epochs = epochs  # Number of training epochs
        self.losses = []  # To store loss at each epoch

    def fit(self, X_train, y_train):
        # Initialize coefficients and intercept
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])  # Initialize coefficients with ones

        for i in range(self.epochs):
            epoch_loss = 0  # Loss for the current epoch
            for j in range(X_train.shape[0]):
                idx = np.random.randint(0, X_train.shape[0])  # Choose a random data point

                # Calculate predicted target value (y_hat) for the chosen data point
                y_hat = np.dot(X_train[idx], self.coef_) + self.intercept_

                # Calculate derivative of the intercept for gradient update
                intercept_der = -1 * (y_train[idx] - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)  # Update intercept using gradient descent

                # Calculate derivative of the coefficients for gradient update
                coef_der = -1 * np.dot((y_train[idx] - y_hat), X_train[idx])
                self.coef_ = self.coef_ - (self.lr * coef_der)  # Update coefficients using gradient descent
                # Calculate and accumulate the loss for the current sample
                epoch_loss += ((y_train[idx] - y_hat)**2)/2

            # Calculate the average loss for the epoch and store it
            self.losses.append(epoch_loss / X_train.shape[0])

        # Print the learned intercept and coefficients after training
        print("Intercept:", self.intercept_)
        print("Coefficients:", self.coef_)

    def predict(self, X_test):
        # Predict target values using learned coefficients and intercept
        return np.dot(X_test, self.coef_) + self.intercept_

Regr =GradientDescent(epochs=1000000,learning_rate=0.0003)
Regr.fit(X_train, y_train)

Regr1=Stoc_Grad_Desc(epochs=200,learning_rate=0.0002)
Regr1.fit(X_train, y_train)

"""# **Q1.5 Plot the loss vs epoch curve.**"""

# Plot the loss vs epoch curve
import matplotlib.pyplot as plt
plt.plot(range(1, Regr.epochs + 1), Regr.losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch Curve GD')
plt.show()

plt.plot(range(1, Regr1.epochs + 1), Regr1.losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch Curve SGD')
plt.show()

"""# **Q1.6. Give a student’s data – [Hours of study = 7, Previous score = 95, Extracurricular Activities =  Yes, Duration of Sleep = 7, Sample Question Papers Practiced = 6] then What will be his/her performance based on your trained model.**"""

Regr.predict([7,95,7,6,1])

Regr1.predict([7,95,7,6,1])

"""# **Q1.7. Evaluate the model’s performance based on any two-performance metrics (at least 2) from below on the test set – a.) MSE error b.) R2 Score c.) Adjusted R2 score**"""

#calculate y_predict from linear regression model(GD)
y_pred=Regr.predict(X_test)

#calculate y_predict from linear regression model(SGD)
#y_pred=Regr1.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = np.mean((y_test - y_pred) ** 2)

# Calculate R2 Score
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual = np.sum((y_test - y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

# Calculate Adjusted R2 Score
n = len(y_test)  # Number of samples
p = 5  # Number of predictors
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

# Create a DataFrame to store the results
metrics_df = pd.DataFrame({'MSE': [mse], 'R2 Score': [r2], 'Adjusted R2': [adjusted_r2]})

print(metrics_df)
