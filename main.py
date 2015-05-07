import os, json
from sys import argv
import pandas as pd
from oauth2client.client import SignedJwtAssertionCredentials
from httplib2 import Http
from googleapiclient.discovery import build


# get API key from environment variable
# api_key = os.getenv('GOOGLE_API_KEY')
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
    result = service.trainedmodels().predict(project='sacred-temple-93605',
                                                 id=model_id,
                                                 body={'input': {'csvInstance': features}}
                                        ).execute()

    return result


def get_testing_data(file_name='data/main_gold_etf_testing.csv'):
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


success_list = [0,0,0] # [correct prediction, incorrect, total guesses]
csv_instance_array = get_testing_data()

# curious to see how well model predicts on out of sample data

features_to_test = csv_instance_array
#print('Actual Label: ' + features_to_test['label'],
#      'Predicted Label: ' + make_prediction(features_to_test['features'])['outputLabel'])
for row in features_to_test:
    actual = row['label']
    predicted = make_prediction(row['features'])['outputLabel']
    if actual == predicted:
        success_list[0] += 1
    else:
        success_list[1] += 1
    success_list[2] += 1

print(success_list)

