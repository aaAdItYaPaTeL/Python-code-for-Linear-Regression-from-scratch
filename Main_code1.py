# -*- coding: utf-8 -*-
"""M22EE051_Question2.ipynb
"""

from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
df = pd.read_csv('student_data.csv')
df=df.drop_duplicates()
# Convert categorical variable to numerical using one-hot encoding
df = pd.get_dummies(df, columns=["Extracurricular Activities"], drop_first=True)
# Split features and target variable
X = df.drop("Performance", axis=1)
y = df["Performance"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

reg = LinearRegression()
reg.fit(X_train,y_train)

print("coff.:",reg.coef_)
print("Intercept:",reg.intercept_)

y_pred =reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
