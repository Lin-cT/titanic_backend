import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class DepressionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.X_test = None
        self.y_test = None
    
    def train_model(self, data_path):
        # Load data
        data = pd.read_csv(data_path)
        
        # Split the data into features and labels
        X = data.drop('Probability of Developing Depression', axis=1)
        y = data['Probability of Developing Depression']

        # Standardize the features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split the data into training and testing sets
        X_train, self.X_test, y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train a linear regression model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
    
    def predict_depression(self, age, stress_level, exercise_hours, sleep_hours):
        if self.model is None or self.scaler is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
        
        input_data = self.scaler.transform([[age, stress_level, exercise_hours, sleep_hours]])
        chance_of_depression = self.model.predict(input_data)[0]
        return chance_of_depression
    
# Usage
depression_model = DepressionModel()

def initDepression():
    depression_model.train_model('depression_dataset.csv')