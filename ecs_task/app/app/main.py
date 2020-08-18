from flask import Flask, request
import boto3
import os
import pickle


class BasicApp:

    def __init__(self):
        self._app = Flask(__name__)
        model_location = os.environ['S3_MODEL_BUCKET']
        model_key = os.environ['S3_MODEL_KEY']
        client = boto3.client('s3')
        # model is downloaded to the key name
        client.download_file(model_location, model_key, model_key)

        with open(model_key, 'rb') as fh:
            self.model = pickle.load(fh)

        @self._app.route('/')
        def ping():
            return "hello"

        @self._app.route('/model')
        def get_model_stats():
            # returns basic stats about the scikit model
            return self.model.get_params()

        @self._app.route('/predict', methods=['POST'])
        def predict():
            prediction = self.model.predict(request.json['input'])
            return {'prediction': str(prediction)}

    def get_app(self):
        return self._app


app = BasicApp().get_app()


if __name__ == "__main__":
    app = BasicApp().get_app()
    app.run()
