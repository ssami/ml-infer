import numpy as np
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput
from bentoml.artifact import SklearnModelArtifact
from sklearn import datasets, linear_model


# from ..train.train import get_data, train_model

# had to copy this from the train script, because either we build the training module
# along with the service code, or we copy the functions as standalone

def get_data():
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test  = diabetes_X[-20:]
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test  = diabetes_y[-20:]

    # pickle.dump(diabetes_X_test[2], open('test.pkl', 'wb'))
    return diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test


def train_model(x_train, x_test, y_train, y_test):
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)

    # The mean square error
    print(f'Mean square error: {np.mean((regr.predict(x_test) - y_test)**2)}')

    # Explained variance score: 1 is perfect prediction
    # and 0 means that there is no linear relationship
    # between X and y.
    print(f'Sample score: {regr.score(x_test, y_test)}')

    return regr

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

        # todo: fix whatever's going on with this input
        return self.artifacts.scikit_model.predict(np.array(j['input']) for j in parsed_json)


if __name__ == "__main__":
    """
    To run this, 
    1. navigate to the parent directory of ml-infer
    2. run  `python -m ml-infer.bento.service ml-infer/bento/service.py`  
    """

    x_train, x_test, y_train, y_test = get_data()
    model = train_model(x_train, x_test, y_train, y_test)

    service = DiabetesRegressor()
    service.pack('scikit_model', model)

    # save locally to build Docker image
    service.save()
    # service.save(base_path='s3://ssami-1577132529')


