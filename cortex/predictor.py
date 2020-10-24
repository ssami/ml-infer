import logging
import pickle
import json
import smart_open
import numpy as np


class PythonPredictor:
    def __init__(self, config):
        self.config = config
        self.model = self.download_model()

    def download_model(self):
        with smart_open.open(self.config['model_location'], 'rb') as fh:
            model = pickle.load(fh)

        logging.info(f"Model loaded successfully from {self.config['model_location']}")

        return model

    def predict(self, payload):
        print(payload['data'])
        prediction = self.model.predict(np.array(payload['data']))
        return json.dumps({'prediction': prediction.tolist()})
