import json
import time
from sys import argv
import pandas as pd
from oauth2client.client import SignedJwtAssertionCredentials
from httplib2 import Http
from googleapiclient.discovery import build


secret_file = argv[1]
with open(secret_file) as ff:
    f = json.loads(ff.read())
    private_key = f['private_key']
    client_email = f['client_email']
    private_key_id = f['private_key_id']
    client_id = f['client_id']


project = 'sacred-temple-93605'
model_id = 'finishedModel' # id of model we want to use on Google Prediction


def make_prediction(features):
    # features is list of features to pass to API

    credentials = SignedJwtAssertionCredentials(client_email, private_key,
                                                    'https://www.googleapis.com/auth/prediction')
    http_auth = credentials.authorize(Http())
    service = build('prediction', 'v1.6', http=http_auth)

    try:
        result = service.trainedmodels().predict(project=project,
                                                id=model_id,
                                                body={'input': {'csvInstance': features}}
                                            ).execute()
        return result
    except Exception as e:
        print(e)

def get_testing_data(file_name='data/main_testing.csv'):
    csv_instance_array = []
    td = pd.read_csv(file_name)
    for row in td.iterrows():
        """ create a feature row in prediction API format
        row[0]: index
        row[1]: data
        row[1][0]: date
        row[1][1]: label
        """
        csv_instance_array.append(
            {'label'   : row[1][1],
                'features': list(row[1][2:])})
    return csv_instance_array

def get_predictions_for_testing_data():
    success_list = [0,0,0] # [correct prediction, incorrect, total guesses]
    csv_instance_array = get_testing_data()

    # curious to see how well model predicts on out of sample data
    features_to_test = csv_instance_array
    for row in features_to_test:
        actual = row['label']
        predicted = make_prediction(row['features'])['outputLabel']
        if actual == predicted:
            success_list[0] += 1
        else:
            success_list[1] += 1
        success_list[2] += 1

    # save results for later
    file_name = 'results_' + str(time.time()) +'.txt'
    with open(file_name, 'w') as f:
        result_str = 'Correct: '+str(success_list[0])+'\nIncorrect: ' + str(success_list[1])+'\nTotal: ' + str(success_list[2])+ '\n'

        f.write(result_str)


#features_to_test = get_testing_data()[10] # arbitrarily testing 10th row

"""
print('Actual Label: ' + features_to_test['label'],
      'Predicted Label: ' + make_prediction(features_to_test['features'])['outputLabel'])
"""
get_predictions_for_testing_data()
