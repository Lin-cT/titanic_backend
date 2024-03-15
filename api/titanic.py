
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
         def post(self):
            try:
                # Get passenger data from the API request
                data = request.get_json() # get the data
                return (jsonify()) #whatever we get function 
              

            except Exception as e:
                return jsonify({'error': str(e)})
            
enc = OneHotEncoder(handle_unknown='ignore') 

# Assuming logreg_model and enc are already defined
api.add_resource(TitanicAPI._Create, '/create', resource_class_args=(logreg_model, enc))

