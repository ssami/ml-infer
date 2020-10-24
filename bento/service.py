import os
import pickle

import boto3
import numpy as np
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput
from bentoml.artifact import SklearnModelArtifact


# dependency thing: https://github.com/bentoml/BentoML/issues/984
@env(conda_channels=["conda-forge"], conda_dependencies=["ruamel.yaml"], auto_pip_dependencies=True)
@artifacts([SklearnModelArtifact('scikit_model')])
class DiabetesRegressor(BentoService):

    @api(input=JsonInput())
    def predict(self, parsed_json):
        # note: I think .pack(), @artifacts and self.artifacts all need to refer to the same model name
        # in this case, "scikit_model"

        # also, parsed json has a list of parsed inputs!
        # https://docs.bentoml.org/en/latest/api/adapters.html?highlight=JsonInput#bentoml.adapters.JsonInput
        # this means that if you are NOT doing batched input, you need to send exactly one input and force to [0]

        inp_data = np.array(parsed_json[0]['input'])
        return self.artifacts.scikit_model.predict(inp_data)


if __name__ == "__main__":
    """
    To run this, 
    1. navigate to the parent directory of ml-infer
    2. run  `python -m ml-infer.bento.service ml-infer/bento/service.py`  
    """

    client = boto3.client('s3')
    # model is downloaded to the key name
    model_location = os.environ['S3_MODEL_BUCKET']
    model_key = os.environ['S3_MODEL_KEY']
    client.download_file(model_location, model_key, model_key)
    model = pickle.load(open(model_key, 'rb'))

    service = DiabetesRegressor()
    service.pack('scikit_model', model)

    # save locally to build Docker image
    service.save()
    # service.save(base_path='s3://ssami-1577132529')


