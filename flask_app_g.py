from collections.abc import Mapping
from flask import Flask, render_template, request, make_response, jsonify, redirect, flash, send_from_directory, session, g
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
model = None
data = None
tokenizer = None
lbl_encoder = None

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
CORS(app, support_credentials=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/init')
def initialize_data():
    global model
    global data
    global tokenizer
    global lbl_encoder
    with open('intents.json') as file:
        data = json.load(file)
    model = keras.models.load_model('chat-model')

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    #load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    return 'initialized'

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
        user_name = request.args.get('u')
        global model
        global data
        global tokenizer
        global lbl_encoder

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]), truncating = 'post', maxlen = max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        log_tag = ''
        response["is_success"] = True
        for i in data['intents']:
            if i['tag'] == tag:
                log_tag = i['tag']
                response["msg"] = np.random.choice(i['responses'])
        response["msg"] = response["msg"].replace("#uname", user_name)
        log(comment["msg"], response["msg"], log_tag)
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

def get_log_path(name, ext):
    path = os.getcwd()
    path_to_save = os.path.join(path, f'{name}.{ext}')
    idx = 0
    while True:
        if os.path.exists(path_to_save):
            idx += 1
        else:
            break
        path_to_save = os.path.join(path, f'{name}_{idx}.{ext}')
        if idx > 10000000:
            path_to_save = os.path.join(path, f'{name}.{ext}')
            os.remove(path_to_save)
            break
    return path_to_save

def log(txt_user, txt_bot, tag):
    log_path = 'chats.log'
    MAX_LOG_SIZE_BYTES = 1 * 1024 * 1024  # 1MB in bytes

    if os.path.exists(log_path) and os.path.getsize(log_path) > MAX_LOG_SIZE_BYTES:
        log_path = get_log_path('chats', 'log')
    
    log_file = open(log_path, 'a')
    msg = f'{datetime.now().strftime("%Y%m%d %H:%M")} - "{txt_user}", "{txt_bot}", "tag: {tag}"\n'
    log_file.write(msg)
    log_file.close()

if __name__ == '__main__':
    # app.run(host="192.168.8.171", port=5000, debug=True)
    # app.run(host="192.168.8.171", port=5000, debug=False, ssl_context='adhoc')
    # app.run(debug=False, host='vulnagent.com/api', port=5000)
    app.run(debug=False, host='192.168.8.171', port=5000)