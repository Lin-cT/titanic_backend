from flask import Blueprint, jsonify, request  # jsonify creates an endpoint response object
from flask_restful import Api, Resource # used for REST API building

from model.jokes import *

joke_api = Blueprint('joke_api', __name__,
                   url_prefix='/api/jokes')

# API generator https://flask-restful.readthedocs.io/en/latest/api.html#id1
api = Api(joke_api)

class TitanicAPI(Resource):
    def post(self):
            # Get passenger data from the API request
            data = request.get_json()  # get the data as JSON
            data['alone'] = str(data['alone']).lower()
            converted_dict = {key: [value] for key, value in data.items()}
            pass_in = pd.DataFrame(converted_dict)  # create DataFrame from JSON
            titanic_predictor = TitanicPredictor()
            titanic_predictor.load_data()
            titanic_predictor.preprocess_data()
            titanic_predictor.train_models()
            titanic_predictor.evaluate_models()
            response = titanic_predictor.predict_survival_probability(pass_in)
            return jsonify(response)

# Add resource to the API
api.add_resource(TitanicAPI, '/create')