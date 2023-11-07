from collections.abc import Mapping
from flask import Flask, render_template, request, make_response, jsonify, redirect, flash, send_from_directory
import sys
import json
from flask_cors import CORS, cross_origin
import jwt
from datetime import datetime, timedelta
from functools import wraps
import uuid
from  werkzeug.security import generate_password_hash, check_password_hash
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import random
import pickle
import os

max_len = 20
SECRET_KEY = "155E@!FAs"

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
CORS(app, support_credentials=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/greet')
def greet():
    name = request.args.get('name', 'Guest')
    return f'Hello, {name}!'

@app.route('/api/chat', methods=['POST'])
@cross_origin(supports_credentials=True)
def chat():
    response = {}
    try:
        comment = request.json
        if not comment["msg"]:
            response["is_success"] = False
            response["msg"] = "No message."

        inp = comment["msg"]
        if inp.lower() == 'quit':
            response["is_success"] = True
            response["msg"] = "Take care. See you soon."
            return response

        with open('intents.json') as file:
            data = json.load(file)

        model = keras.models.load_model('chat-model')

        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        #load label encoder object
        with open('label_encoder.pickle', 'rb') as enc:
            lbl_encoder = pickle.load(enc)

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]), truncating = 'post', maxlen = max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        response["is_success"] = True
        for i in data['intents']:
            if i['tag'] == tag:
                response["msg"] = np.random.choice(i['responses'])

        return response
    except Exception as ee:
        response["is_success"] = False
        response["msg"] = ee
    return response

@app.errorhandler(403)
def forbidden(e):
    return jsonify({
        "message": "Forbidden",
        "error": str(e),
        "data": None
    }), 403

@app.errorhandler(404)
def forbidden(e):
    return jsonify({
        "message": "Endpoint Not Found",
        "error": str(e),
        "data": None
    }), 404
