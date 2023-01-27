import re
from flask import Flask, request, send_from_directory
from predictor import get_predictions
from flask_cors import CORS, cross_origin
import os

PREFIX_REGEX = r'^[a-zA-Z]+$'
MAX_AMOUNT = 10
DEFAULT_AMOUNT = 5
DEFAULT_TOP = 5
MAX_TOP = 27

app = Flask(__name__, static_folder='front')
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/predict', methods=['POST'])
@cross_origin()
def predict():
    prefix = request.args.get('prefix')
    if prefix is not None and not re.match(PREFIX_REGEX, prefix):
        return 'Invalid prefix', 400

    amount = request.args.get('amount', '5')
    if not amount.isdecimal():
        return 'Invalid amount', 400
    else:
        amount = int(amount)

    top = request.args.get('top', '10')
    if not top.isdecimal():
        return 'Invalid top', 400
    else:
        top = int(top)

    max_len = request.args.get('maxLen', '24')
    if not max_len.isdecimal():
        return 'Invalid maxLen', 400
    else:
        max_len = int(max_len)

    uniform = bool(request.args.get('uniform'))

    if prefix is not None and len(prefix) >= max_len:
        return 'Input query was to long!', 400

    if amount > MAX_AMOUNT or amount <= 0:
        return f'"amount" out of range: [1; {MAX_AMOUNT}]', 400
    if max_len <= 0:
        return '"maxLen" out of range: [1; )', 400
    if top <= 0 or top > MAX_TOP:
        return f'"top" out of range: [1, {MAX_TOP}', 400

    return get_predictions(prefix, amount, max_len, uniform, top)
