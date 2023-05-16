from flask import Flask, jsonify, request
import json
import sys
import numpy as np
from utils.mesh_clasification_data_utils import predictJawClass
from utils.mesh_segmentation_data_utils import get_tooth_segmentation_vertices, get_trim_gum_vertices


app = Flask(__name__)
print("Server running ...")
@app.route("/ping", methods=["POST"])
def api_1():
    print("Ping successful!")

    try:
        request_json = request.get_json()
        response = jsonify(request_json)
        response.status_code = 200
    except:
        exception_message = sys.exc_info()[1]
        response = json.dumps({"content": exception_message})
        response.status_code = 400
    return response


@app.route("/api_classifier", methods=["POST"])
def api_classifier():
    print("Ping successful!")
    try:
        # request_json = request.get_json()
        data = request.get_json()
    except:
        raise ValueError("Input can't be parsed!")
        exception_message = sys.exc_info()[1]
    # data = request_json
    jawClass = predictJawClass(data["enhanced"], model_dir='models/EnhancedBinaryJawClassification')
    jaw_desc = {0: 'Gips Lower Jaw', 1: 'Gips Upper Jaw', 2: 'IO Lower Jaw', 3: 'IO Upper Jaw'}
    print(f'predicted class: "{jawClass[0]}" : {jaw_desc[jawClass[0]]}')
    trimmed_gum, point_index, original_points = get_trim_gum_vertices(data["original"], jawClass[0])
    print('max predicted labels:', np.max(trimmed_gum))
    print('predicted labels len:', len(trimmed_gum))
    print('point indexes:', point_index[:10])
    print('point indexes len:', len(point_index))
    result = {"jawClass": int(jawClass[0]), "predictions": trimmed_gum.tolist(), "pointIndexes": point_index.tolist(), "points": original_points.tolist()}
    response = jsonify(result)
    print('AI Server Task Done!')
    return response


@app.route("/api_upper", methods=["POST"])
def api_upper():
    print("Ping successful!")
    try:
        # request_json = request.get_json()
        data = request.get_json()
    except:
        raise ValueError("Input can't be parsed!")
        exception_message = sys.exc_info()[1]
    # data = request_json
    print("Recieved data: ", data[:3])
    pred_val, point_index = get_tooth_segmentation_vertices(data, model_dir='models/orcalign_trimmed_ai_upper', jaw="upper")
    print('predicted labels:', pred_val[:3])
    print('predicted labels len:', len(pred_val))
    print('point indexes:', point_index[:3])
    print('point indexes len:', len(point_index))

    response = jsonify({"predictions": pred_val.tolist(), "pointIndexes": point_index.tolist()})
    print('AI Server Task Done!')
    return response

@app.route("/api_lower", methods=["POST"])
def api_lower():
    print("Ping successful!")
    try:
        # request_json = request.get_json()
        data = request.get_json()
    except:
        raise ValueError("Input can't be parsed!")
        exception_message = sys.exc_info()[1]
    # data = request_json
    print("Recieved data: ", data[:3])
    pred_val, point_index = get_tooth_segmentation_vertices(data, model_dir='models/orcalign_trimmed_ai_lower', jaw="lower")
    print('predicted labels:', pred_val[:3])
    print('predicted labels len:', len(pred_val))
    print('point indexes:', point_index[:3])
    print('point indexes len:', len(point_index))

    response = jsonify({"predictions": pred_val.tolist(), "pointIndexes": point_index.tolist()})
    print('AI Server Task Done!')
    return response

if __name__ == "__main__":
    # from waitress import serve
    app.run(host='0.0.0.0', port='7000', debug=False)
    # app.run(debug=False)
    # serve(app, host="0.0.0.0", port=7000)