import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

def initTitanic():
    titanic_data = sns.load_dataset('titanic')
    return titanic_data

def preprocess_passenger_data(passenger):
    # Preprocessing code for Titanic data
    passenger['sex'] = passenger['sex'].apply(lambda x: 1 if x == 'male' else 0)
    passenger['alone'] = passenger['alone'].apply(lambda x: 1 if x else 0)

    # Encode 'embarked' variable
    enc = OneHotEncoder(drop='first', sparse=False)
    onehot = enc.fit_transform(passenger[['embarked']])
    cols = ['embarked_' + val for val in enc.get_feature_names(['embarked'])]
    passenger[cols] = pd.DataFrame(onehot, index=passenger.index)

    # Drop unnecessary columns
    passenger.drop(['name', 'embarked'], axis=1, inplace=True)

    return passenger

def train_logistic_regression_model(X_train, y_train):
    # Initialize and train the logistic regression model
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    return logreg

def predict_survival_probability(logreg, new_passenger):
    # Your prediction code for Titanic survival probability
    dead_proba, alive_proba = np.squeeze(logreg.predict_proba(new_passenger))
    return dead_proba, alive_proba

def main():
    # Load the Titanic dataset
    titanic_data = initTitanic()

    # Assuming 'survived' is the target variable
    X = titanic_data.drop('survived', axis=1)
    y = titanic_data['survived']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the logistic regression model
    logreg = train_logistic_regression_model(X_train, y_train)

    # Test the model on the testing set
    y_pred = logreg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy:.2%}')

    return logreg

if __name__ == "__main__":
    logreg_model = main()
