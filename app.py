from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib
import sklearn

model = joblib.load("models/model.joblib")

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['1']
    data2 = request.form['2']
    data3 = request.form['3']
    data4 = request.form['4']
    data5 = request.form['5']
    data6 = request.form['6']
    data7 = request.form['7']
    data8 = request.form['8']
    data9 = request.form['9']
    data10 = request.form['10']
    data11 = request.form['11']

    X = np.array([[data1, data2, data3, data4, data5, data6, 
                   data7, data8, data9, data10, data11]])
    pred = model.predict(X)
    return render_template('prediction.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)