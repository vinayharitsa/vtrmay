import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


@app.route('/prediction/exp/<int:exp>/test/<int:test>/interview/<int:interview>', methods=['GET'])
def prediction(exp, test, interview):
    # userid = prediction.find('userid')
    # test = prediction.find('test')
    # interview = prediction.find('interview')
    final_features = [np.array(exp, test, interview)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    res = json.dumps(output, cls=JSONEncoder)
    return Response(res, mimetype='application/json')


if __name__ == "__main__":
    app.run(debug=True)
