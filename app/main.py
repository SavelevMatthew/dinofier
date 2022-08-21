from flask import Flask, request, jsonify, render_template
from utils import get_prediction

app = Flask(__name__)


@app.route('/names', methods=['POST'])
def predict():
    prefix = request.args.get('startsWith')
    amount = request.args.get('amount')

    return jsonify(get_prediction(prefix, amount))


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
