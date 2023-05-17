import requests
import cv2
import numpy as np
import base64
import json
import logging
logging.basicConfig(level=logging.DEBUG)
from logging.config import dictConfig

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})


def ping(url='http://localhost:5000/ping'):
    res = requests.get(url)
    print('response from server:', res.status_code)
    return res


def postTestImage(url='http://localhost:5000/predict', testImagePath = "static/images/test.jpeg"):
    with open(testImagePath, "rb") as f:
        im_bytes = f.read()
    im_b64 = base64.b64encode(im_bytes).decode("utf8")
    print("Base 64 encoded image first 100 strings: ", im_b64[:100])
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    payload = {"image": im_b64}
    # payload = im_b64
    print("Payload last 100 strings: ", payload['image'][-100:])

    # json format: (b'{"image": "/9j/4AAQSkZJRgABAQ
    # data format: (b'image=%2F9j%2F4AAQSkZJRgABAQAAAQ
    response = requests.post(url, data=payload, headers=headers)
    print("server responsed ... ", response)
    try:
        data = response.json()
        print(data['3D Face ply'])
        logging.info("data propagated successfully!")
    except requests.exceptions.RequestException:
        print(response.status_code)

def postTestImageMultiDict(url='http://localhost:5000/predict3d', testImagePath = "static/images/test.jpeg"):
    """ to send a request in html url-encoded MultiDict format """
    from werkzeug.datastructures import ImmutableMultiDict
    with open(testImagePath, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        req_file = encoded_image.decode('utf-8')

    headers = {'Content-type': 'application/x-www-form-urlencoded', 'Accept': 'text/html'}
    payload = ImmutableMultiDict([('image', req_file)])

    response = requests.post(url, data=payload, headers=headers) # form: for html form, data: for local data

    print('request sent!', response.json().keys())
    try:
        data = response.json()
        print(data['face3d_ply'][:1000])
        print(data['landmark_3d'])
        logging.info("data propagated successfully!")
        logging.info("  ===== FINISHED! =====   ")
    except requests.exceptions.RequestException:
        print(response.status_code)


if __name__ == "__main__":
    # ping = ping()
    ping = ping(url='http://dev1.orcadent.de:9425/ping')
    if ping.status_code == 200:
        logging.info("Send image ...")
        # postTestImage()
        # postTestImageMultiDict()
        postTestImageMultiDict(url='http://dev1.orcadent.de:9425/predict3d')
        # postTestImage(url='http://dev1.orcadent.de:9421/predict')
    else:
        print(f"Connection failed mit status code : {ping.status_code}")
