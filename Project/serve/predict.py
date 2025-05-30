#!/usr/bin/env python
import argparse
import json
import os
import pickle
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from model import LSTMClassifier

from utils import review_to_words, convert_and_pad

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'text/plain':
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

def predict_fn(input_data, model):
    print('Inferring sentiment of input data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model.word_dict is None:
        raise Exception('Model has not been loaded properly, no word_dict.')
    
    # Step 1: Preprocess the input
    cleaned_review = review_to_words(input_data)
    data_X, data_len = convert_and_pad(model.word_dict, cleaned_review)

    # Step 2: Combine length and padded review into a single array
    data_pack = np.hstack(([data_len], data_X)).reshape(1, -1)
    data = torch.from_numpy(data_pack).long().to(device)

    # Step 3: Put model in evaluation mode and get prediction
    model.eval()
    with torch.no_grad():
        output = model(data)

    # Step 4: Convert model output (logits) to prediction (0 or 1)
    result = torch.round(output.squeeze()).item()
    result = int(result)

    return result

from flask import Flask, request, jsonify

app = Flask(__name__)
model = model_fn('/opt/ml/model')  # SageMaker mount model ở đây

@app.route('/ping', methods=['GET'])
def ping():
    return 'pong', 200

@app.route('/invocations', methods=['POST'])
def invoke():
    input_data = input_fn(request.data, 'text/plain')
    result = predict_fn(input_data, model)
    return output_fn(result, 'application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    