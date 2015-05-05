import os, json
from oauth2client.client import SignedJwtAssertionCredentials
from httplib2 import Http
from googleapiclient.discovery import build


# get API key from environment variable
# api_key = os.getenv('GOOGLE_API_KEY')
# TODO: make environment variables or filename as cmd line arg
filename= "stockprediction-66c922ad1bb7.json"
with open(filename) as ff:
    f = json.loads(ff.read())
    private_key = f['private_key']
    client_email = f['client_email']
    private_key_id = f['private_key_id']
    client_id = f['client_id']



project ='sacred-temple-93605'

credentials = SignedJwtAssertionCredentials(client_email, private_key,
                                            'https://www.googleapis.com/auth/sqlservice.admin')
http_auth = credentials.authorize(Http())

service = build('prediction', 'v1.6', http=http_auth)
