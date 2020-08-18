from sklearn import linear_model, datasets
import numpy as np
import pickle
import boto3

S3_MODEL_LOC = '1577132529'

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test  = diabetes_X[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test  = diabetes_y[-20:]

pickle.dump(diabetes_X_test[2], open('test.pkl', 'wb'))

regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)

# The mean square error
print(f'Mean square error: {np.mean((regr.predict(diabetes_X_test) - diabetes_y_test)**2)}')

# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and y.
print(f'Sample score: {regr.score(diabetes_X_test, diabetes_y_test)}')

# save the model for upload/use later
model_file_name = 'diabetes-regression.pkl'
with open(model_file_name, 'wb') as fb:
    pickle.dump(regr, fb)

with open(model_file_name, 'rb') as fb:
    # upload the object into an S3 bucket
    # this assumes you have all the necessary privileges!
    client = boto3.client('s3')
    client.put_object(
        Bucket=S3_MODEL_LOC,
        Body=fb,
        Key=model_file_name
    )
    print(f'Successfully uploaded to {S3_MODEL_LOC}')

