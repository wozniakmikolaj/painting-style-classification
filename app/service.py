import os
import traceback

from flask import Flask, jsonify, request

from src.executors.skip_con_predictor import CNNSkipConnectionPredictor

app = Flask(__name__)

APP_ROOT = os.getenv('APP_ROOT', '/classify')
HOST = "0.0.0.0"
PORT_NUMBER = int(os.getenv('PORT_NUMBER', 8080))

skip_con_predictor = CNNSkipConnectionPredictor()


@app.route(APP_ROOT, methods=["POST"])
def infer():
    data = request.json
    image = data['image']
    return skip_con_predictor.infer(image)


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify(stackTrace=traceback.format_exc())


if __name__ == '__main__':
    app.run(host=HOST, port=PORT_NUMBER)
