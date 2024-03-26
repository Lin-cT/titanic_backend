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
    """A class created to predict the liklihood of depression based off of daily lifestyle
    """

    # a singleton instance of TitanicModel, created to train the model only once, while using it for prediction multiple times
    _instance = None

    # an instance to initialize the DepressionModel
    def __init__(self):
        # the Depression ML Model
        self.model = None
        self.scaler = None
        self.X_test = None
        self.y_test = None
    
    # function to train the model using linear regression
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
    
    # function to predict the chance of depression
    def predict_depression(self, age, stress_level, exercise_hours, sleep_hours):
        """
        Predict the probability of depression for an individual.

        Args:
        age (float): The age of the individual.
        stress_level (float): The stress level of the individual.
        exercise_hours (float): The number of hours the individual exercises per week.
        sleep_hours (float): The number of hours the individual sleeps per night.

        Returns:
        dict: A dictionary containing probabilities of experiencing depression
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
        
        input_data = self.scaler.transform([[age, stress_level, exercise_hours, sleep_hours]])
        chance_of_depression = self.model.predict(input_data)[0]
        return chance_of_depression
    
    @classmethod
    def get_instance(cls):
        """ 
        This function retrieves and constructs the singleton instance of the DepressionModel.
        The DepressionModel is employed for analyzing depression-related data and making predictions or assessments.

        The DepressionModel does not need to be cleaned, as all data send in is already cleaned and there are no string values.

        Returns:
            DepressionModel: The singleton instance of the DepressionModel, encompassing both data and prediction methods.
        """
        # check for instance, if it doesn't exist, create it
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.train_model('depression_dataset.csv')
        # return the instance, to be used for prediction
        return cls._instance

# Usage
depressionModel = DepressionModel()

def initDepression():
    """ Initialize the Depression Model.
    This function is used to load the Depression Model into memory, and prepare it for prediction.
    """
    depressionModel.get_instance()

def testDepression():
    """ Test the Depression Model
    Using the DepressionModel class, we can predict the likelihood of depression based on daily lifestyle.
    Print output of this test contains method documentation, individual data, and depression probability.
    """
    # Setup data for prediction
    print(" Step 1: Define individual data for prediction:")
    individual_data = {
        'age': 30,
        'stress_level': 5,
        'exercise_hours': 3,
        'sleep_hours': 7
    }

    age = 30
    stress_level = 5
    exercise_hours = 3
    sleep_hours = 7

    print("\t", individual_data)
    print()

    # Get an instance of the trained Depression Model
    depressionModel = DepressionModel.get_instance()
    print(" Step 2:", depressionModel.get_instance.__doc__)

    # Predict the probability of depression
    print(" Step 3:", depressionModel.predict_depression.__doc__)
    depression_probability = depressionModel.predict_depression(age, stress_level, exercise_hours, sleep_hours)
    print('\t Probability of depression: {:.2%}'.format(depression_probability))
    print()

if __name__ == "__main__":
    print(" Begin:", testDepression.__doc__)
    testDepression()
