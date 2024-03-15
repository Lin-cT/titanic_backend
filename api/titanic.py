
from model.titanics import *
from flask import Blueprint, request, jsonify, current_app, Response
from flask_restful import api, Resource # used for REST API building
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder



titanic_api = Blueprint('titanic_api', __name__,
                   url_prefix='/api/titanic')


class TitanicAPI(Resource):
    class _Create(Resource):
        def __init__(self, logreg, enc):
            super().__init__()
            self.logreg = logreg
            self.enc = enc

        def post(self):
            try:
                # Get passenger data from the API request
                data = request.get_json()
                passenger_data = pd.DataFrame([data])

                # Preprocess the passenger data
                passenger_data['sex'] = passenger_data['sex'].apply(lambda x: 1 if x == 'male' else 0)
                passenger_data['alone'] = passenger_data['alone'].apply(lambda x: 1 if x else 0)
                onehot = self.enc.transform(passenger_data[['embarked']]).toarray()
                cols = ['embarked_' + val for val in self.enc.categories_[0]]
                passenger_data[cols] = pd.DataFrame(onehot, index=passenger_data.index)
                passenger_data.drop(['name', 'embarked'], axis=1, inplace=True)

                # Predict the survival probability
                dead_proba, alive_proba = np.squeeze(self.logreg.predict_proba(passenger_data))

                # Return the results in JSON format
                response = {
                    'death_probability': float(dead_proba),
                    'survival_probability': float(alive_proba)
                }

                return jsonify(response)

            except Exception as e:
                return jsonify({'error': str(e)})
            
enc = OneHotEncoder(handle_unknown='ignore') 

# Assuming logreg_model and enc are already defined
api.add_resource(TitanicAPI._Create, '/create', resource_class_args=(logreg_model, enc))

