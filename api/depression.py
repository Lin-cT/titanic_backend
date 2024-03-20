from flask import Blueprint, request, jsonify, make_response
from flask_restful import Api, Resource
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

predict_api = Blueprint("predict_api", __name__,
                        url_prefix='/api/predict')
api = Api(predict_api)
# Load the dataset
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

class Predict(Resource):
    def post(self):
        body = request.get_json()
        age = float(body.get("age"))
        stress_level = float(body.get("stress"))
        daily_exercise_hours = float(body.get("exercise"))
        daily_sleep_hours = float(body.get("sleep"))
        chance_of_depression = predict_depression(age, stress_level, daily_exercise_hours, daily_sleep_hours)
        chance_of_depression = max(0, min(chance_of_depression, 1))  # Ensure chance_of_depression is between 0 and 1
        return jsonify(f"Based on the provided data, the chance of developing depression is: {chance_of_depression * 100:.2f}%")

api.add_resource(Predict, '/')
