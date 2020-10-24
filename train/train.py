from sklearn import linear_model, datasets
import numpy as np
import pickle
import boto3

S3_MODEL_LOC = '1577132529'


def get_data():
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test  = diabetes_X[-20:]
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test  = diabetes_y[-20:]

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


def save_model(model, file_name, remote_loc):
    # save the model for upload/use later
    with open(file_name, 'wb') as fb:
        pickle.dump(model, fb)

    with open(file_name, 'rb') as fb:
        # upload the object into an S3 bucket
        # this assumes you have all the necessary privileges!
        client = boto3.client('s3')
        client.put_object(
            Bucket=remote_loc,
            Body=fb,
            Key=file_name
        )
        print(f'Successfully uploaded to {remote_loc}')


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_data()
    model = train_model(x_train, x_test, y_train, y_test)
    save_model(model, 'diabetes_regression.pkl', S3_MODEL_LOC)
