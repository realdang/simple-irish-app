from flask import Flask
from flask_cors import CORS, cross_origin
from flask import render_template
from flask import request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')

def home():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])

@cross_origin(origin='*')

def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text ="{0}".format(prediction))
    
if __name__== '__main__':
    app.run(debug=True)