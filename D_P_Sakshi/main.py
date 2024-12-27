import numpy as np
import pickle
from flask import Flask, jsonify, render_template, request, redirect

app = Flask(__name__)

def predict(values, dic):
    if len(values) == 8:
        model = pickle.load(open('covid.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 15:
        model = pickle.load(open('lung_cancer.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 11:
        model = pickle.load(open('heart.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/lung',methods=['GET','POST'])
def lung():
  return render_template('Lung_cancer.html')

@app.route('/covid',methods=['GET','POST'])
def covid():
    return render_template('covid.html')

@app.route('/heart',methods=['GET','POST'])
def heart():
  return render_template("Heart_disease.html")

@app.route('/predict',methods=['GET','POST'])
def predictT():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except:
        message = "Entered unexpected data, please enter again"
        return render_template("home.html", message = message)

    return render_template('predict.html', pred = pred)




if __name__=='__main__':
  app.run(debug=True)