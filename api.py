import numpy as np

#from parse import *
import requests
import base64
import json
import cv2

with open('tokens.json', 'r') as f:
    tokens = json.load(f)
    OAUTH_TOKEN = tokens['OAUTH_TOKEN']
    CATALOG_ID = tokens['CATALOG_ID']
    RECOGNITION_URL = tokens['RECOGNITION_URL']


img_path = 'test_images/test_img.jpg'


def get_response(img_path='test_images/test_img.jpg', save_response_json=True):
    # Getting IAM-token
    try:
        response = requests.post(
            "https://iam.api.cloud.yandex.net/iam/v1/tokens",
            json={"yandexPassportOauthToken": OAUTH_TOKEN},
        )
    except requests.exceptions.ConnectionError:
        print('Getting IAM-token failed: ConnectionError')


    IAM_TOKEN = json.loads(response.text)['iamToken']

    # Getting an image
    img = open(img_path, 'rb').read()
    img_cv = cv2.imread(img_path)
    encoded_img = base64.b64encode(img)

    # Data dor request
    data = json.dumps({
        "mimeType": 'JPEG',
        "languageCodes": ["*"],
        "model": 'page',
        "content": encoded_img.decode()
    })
    # Headers for request
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + IAM_TOKEN,
        "x-folder-id": CATALOG_ID,
        "x-data-logging-enabled": "true"
    }
    # Respnose to Yandex Vsion API
    response = requests.post(RECOGNITION_URL, headers=headers, data=data)
    result = json.loads(response.content)['result']
    if save_response_json:
        with open('test_jsons/' + img_path.split('/')[-1] + '.json', 'w') as f:
            json.dump(result, f)
    return result
