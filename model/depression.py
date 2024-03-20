import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('depression_dataset.csv')

# Split the data into features and labels
X = data.drop('Probability of Developing Depression', axis=1)
y = data['Probability of Developing Depression']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Function to predict the chance of being depressed
def predict_depression(age, stress_level, exercise_hours, sleep_hours):
    input_data = scaler.transform([[age, stress_level, exercise_hours, sleep_hours]])
    chance_of_depression = model.predict(input_data)[0]
    return chance_of_depression