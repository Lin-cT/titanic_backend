from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

player_api = Blueprint('player_api', __name__,
                   url_prefix='/api/players')

# API docs https://flask-restful.readthedocs.io/en/latest/api.html
api = Api(player_api)

# Load the dataset
data = pd.read_csv('depression_dataset.csv')

# Dropping 'Family History of Depression' from features
data.drop('Family History of Depression', axis=1, inplace=True)

# Features and labels
X = data.drop('Depression', axis=1)
y = data['Depression']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Function to predict the probability of depression
def predict_depression(age, stress_level, exercise_hours, sleep_hours):
    input_data = scaler.transform([[age, stress_level, exercise_hours, sleep_hours]])
    probability_of_depression = model.predict_proba(input_data)[:, 1][0]
    return probability_of_depression

# Take user input
class Predict(Resource):
    def post(self):
        body = request.get_json()
        age = float(body.get("age"))
        stress_level = float(body.get("stress_level"))
        exercise_hours = float(body.get("exercise_hours"))
        sleep_hours = float(body.get("sleep_hours"))
        probability_of_depression = predict_depression(age, stress_level, exercise_hours, sleep_hours)
        return jsonify({"message": f"Based on the provided data, the probability of depression is: {probability_of_depression * 100:.2f}%"})

api.add_resource(Predict, '/')
