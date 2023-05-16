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
        # print(response.content)
        logging.info("data propagated successfully!")
    except requests.exceptions.RequestException:
        print(response.status_code)

    # converting base64 string to numpy array
    # orgImageStr= response.json()['alignedFace']
    # img_data = base64.b64decode(orgImageStr)
    # nparr = np.fromstring(img_data, np.uint8)
    # img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    image_name = testImagePath.split('.')[0].split('/')[-1]
    original_image = cv2.imread(testImagePath, cv2.IMREAD_COLOR)

    # Transparent Mouth
    transparent_mouth = response.json()['transparentMouth']
    img_data_t = base64.b64decode(transparent_mouth)
    nparr = np.fromstring(img_data_t, np.uint8)
    img_tp = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    # cv2.imwrite(f'static/results/{image_name}_original_image_transparent.png', img_tp,
    #             [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

    affine_matrix = json.loads(response.json()['affine_matrix'])
    affine_matrix = np.array(affine_matrix)

    points = json.loads(response.json()['original_points'])
    original_pointed = original_image.copy()
    for (x, y) in points:
        cv2.circle(original_pointed, (x, y), 10, (255, 255, 55), 10)
    # cv2.imwrite(f'static/results/{image_name}_original_image_pointed.png', original_pointed)

    # Affine transformation of pointed image using calculated rotations matrix
    transformed_image = cv2.warpAffine(original_pointed, affine_matrix,
                                       (original_image.shape[1], original_image.shape[0]), flags=cv2.INTER_CUBIC)
    # cv2.imwrite(f'static/results/{image_name}_transformed_image_pointed.png', transformed_image)

    # Affine transformation of transparent mouth image using calculated rotations matrix
    transformed_transparent = cv2.warpAffine(img_tp, affine_matrix, (img_tp.shape[1], img_tp.shape[0]),
                                             flags=cv2.INTER_CUBIC)
    # cv2.imwrite(f'static/results/{image_name}_transformed_image_transparent.png', transformed_transparent,
    #             [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

    # local test image visualization

    cv2.imshow('Detected 70 Points',
               cv2.resize(original_pointed, (int(original_image.shape[1] * 0.2), int(original_image.shape[0] * 0.2))))
    cv2.imshow('Transparent Image', cv2.resize(img_tp, (int(img_tp.shape[1] * 0.2), int(img_tp.shape[0] * 0.2))))
    cv2.imshow('Aligned Pointed Image', cv2.resize(transformed_image, (
    int(transformed_image.shape[1] * 0.2), int(transformed_image.shape[0] * 0.2))))
    cv2.imshow('Aligned Transparent Image', cv2.resize(transformed_transparent, (
    int(transformed_transparent.shape[1] * 0.2), int(transformed_transparent.shape[0] * 0.2))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def postTestImageMultiDict(url='http://localhost:5000/predict', testImagePath = "static/images/test.jpeg"):
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
        # print('response: ', response.content)
        logging.info("data propagated successfully!")
    except requests.exceptions.RequestException:
        print(response.status_code)

    # converting base64 string to numpy array
    # orgImageStr= response.json()['alignedFace']
    # img_data = base64.b64decode(orgImageStr)
    # nparr = np.fromstring(img_data, np.uint8)
    # img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    image_name = testImagePath.split('.')[0].split('/')[-1]
    original_image = cv2.imread(testImagePath, cv2.IMREAD_COLOR)

    # Transparent Mouth
    transparent_mouth = response.json()['transparentMouth']
    img_data_t = base64.b64decode(transparent_mouth)
    nparr = np.frombuffer(img_data_t, np.uint8)
    img_tp = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    # cv2.imwrite(f'static/results/{image_name}_original_image_transparent.png', img_tp,
    #             [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

    affine_matrix = response.json()['affine_matrix']
    affine_matrix = np.array(affine_matrix)

    points = response.json()['original_points']
    original_pointed = original_image.copy()
    for (x, y) in points:
        cv2.circle(original_pointed, (x, y), 10, (255, 255, 55), 10)
    # cv2.imwrite(f'static/results/{image_name}_original_image_pointed.png', original_pointed)

    # Affine transformation of pointed image using calculated rotations matrix
    transformed_image = cv2.warpAffine(original_pointed, affine_matrix,
                                       (original_image.shape[1], original_image.shape[0]), flags=cv2.INTER_CUBIC)
    # cv2.imwrite(f'static/results/{image_name}_transformed_image_pointed.png', transformed_image)

    # Affine transformation of transparent mouth image using calculated rotations matrix
    transformed_transparent = cv2.warpAffine(img_tp, affine_matrix, (img_tp.shape[1], img_tp.shape[0]),
                                             flags=cv2.INTER_CUBIC)
    # cv2.imwrite(f'static/results/{image_name}_transformed_image_transparent.png', transformed_transparent,
    #             [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

    # local test image visualization

    cv2.imshow('Detected 70 Points',
               cv2.resize(original_pointed, (int(original_image.shape[1] * 0.2), int(original_image.shape[0] * 0.2))))
    cv2.imshow('Transparent Image', cv2.resize(img_tp, (int(img_tp.shape[1] * 0.2), int(img_tp.shape[0] * 0.2))))
    cv2.imshow('Aligned Pointed Image', cv2.resize(transformed_image, (
    int(transformed_image.shape[1] * 0.2), int(transformed_image.shape[0] * 0.2))))
    cv2.imshow('Aligned Transparent Image', cv2.resize(transformed_transparent, (
    int(transformed_transparent.shape[1] * 0.2), int(transformed_transparent.shape[0] * 0.2))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    ping = ping()
    # ping = ping(url='http://dev1.orcadent.de:9421/ping')
    if ping.status_code == 200:
        logging.info("Send image ...")
        # postTestImage()
        postTestImageMultiDict()
        # postTestImageMultiDict(url='http://dev1.orcadent.de:9421/predict')
        # postTestImage(url='http://dev1.orcadent.de:9421/predict')
    else:
        print(f"Connection failed mit status code : {ping.status_code}")
