import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt 
import time
import pandas as pd
import os
from flask import Flask , request, render_template
#from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model_path = r'E:\Datas\VIT_Morning_Slot-main\VIT_Morning_Slot-main\cnn_flask\uploads\model1.pkl'

try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print(f"Error: File not found at '{model_path}'")
scale= pickle.load(open(r'E:\Datas\VIT_Morning_Slot-main\VIT_Morning_Slot-main\cnn_flask\uploads\scale.pkl','rb'))
                 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    input_feature=[float(x) for x in request.form.values()]
    features_value=[np.array(input_feature)]
    names = [['holiday','temp','rain','snow','weather','year','month','day','hours','minutes','seconds']]
    data = pd.DataFrame(features_value,columns=names)
    prediction = model.predict(data)
    data=pd.DataFrame(data,columns=names)
    prediction=model.predict(data)
    print(prediction)
    text="Estimated Traffic Volume is: "
    return render_template("index.html",prediction_text = text + str(prediction))

if __name__ == '__main__':
    port = int(os.environ.get('PORT',5000))
    app.run(debug = False, threaded = False)
