import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the red wine dataset
red_wine_data = pd.read_csv("winequality-red.csv", sep=";")

# Load the white wine dataset
white_wine_data = pd.read_csv("winequality-white.csv", sep=";")

# Add a 'wine_type' column to distinguish between red and white wines
red_wine_data['wine_type'] = 'red'
white_wine_data['wine_type'] = 'white'

# Combine the datasets into a single DataFrame
combined_data = pd.concat([red_wine_data, white_wine_data])

# Encode the 'wine_type' column to numeric values
le = LabelEncoder()
combined_data['wine_type'] = le.fit_transform(combined_data['wine_type'])

# Define the features (X) and target variable (y)
X = combined_data.drop('quality', axis=1)
y = combined_data['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (excluding 'wine_type')
scaler = StandardScaler()
X_train[['wine_type']] = scaler.fit_transform(X_train[['wine_type']])
X_test[['wine_type']] = scaler.transform(X_test[['wine_type']])

# Build a Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Get the feature names
feature_names = X.columns

# Get the coefficients of the linear regression model
coefficients = lr.coef_

# Print the coefficients for each feature
print("Coefficients for each feature:")
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")

# Print model evaluation metrics
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# Plot feature coefficients
plt.figure(figsize=(10, 6))
plt.title("Feature Coefficients")
plt.bar(range(len(coefficients)), coefficients, align="center")
plt.xticks(range(len(coefficients)), feature_names, rotation=45)
plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.tight_layout()
plt.show()
