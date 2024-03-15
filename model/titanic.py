import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class TitanicRegression:
    def __init__(self):
        self.dt = DecisionTreeClassifier()
        self.logreg = LogisticRegression()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.encoder = OneHotEncoder()

    def initTitanic(self):
        titanic_data = sns.load_dataset('titanic')
        # Clean data (handle missing values, encode categorical variables, etc.)
        # For example:
        titanic_data.dropna(inplace=True)
        self.X = titanic_data.drop('survived', axis=1)  # Features
        self.y = titanic_data['survived']  # Target variable
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)
        self.encoder.fit(self.X_train)  # Fit the encoder on training data
        self.X_train = self.encoder.transform(self.X_train)  # Transform training data
        self.X_test = self.encoder.transform(self.X_test)  # Transform test data

    def runDecisionTree(self):
        self.dt.fit(self.X_train, self.y_train)

    def runLogisticRegression(self):
        self.logreg.fit(self.X_train, self.y_train)

    def predictSurvival(self, passenger):
        encoded_passenger = self.encoder.transform(passenger)
        dt_prediction = self.dt.predict(encoded_passenger)
        logreg_prediction = self.logreg.predict(encoded_passenger)
        # Combine or select one of the predictions as the final prediction
        return dt_prediction, logreg_prediction

if __name__ == "__main__":
    titanic_regression = TitanicRegression()
    titanic_regression.initTitanic()
    titanic_regression.runDecisionTree()
    titanic_regression.runLogisticRegression()
    # Now you can use predictSurvival method to make predictions
