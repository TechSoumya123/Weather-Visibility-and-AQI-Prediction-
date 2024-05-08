import pickle

import numpy as np
from flask import Flask, render_template, request

model = pickle.load(open('RandomModel.pkl', 'rb'))

vimodel = pickle.load(open('visibility.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict_user():
    T = float(request.form['T'])
    TM = float(request.form['TM'])
    Tm = float(request.form['Tm'])
    SLP = float(request.form['SLP'])
    H = int(request.form['H'])
    VV = float(request.form['VV'])
    V = float(request.form['V'])
    VM = float(request.form['VM'])
    features = np.array([[T, TM, Tm, SLP, H, VV, V, VM]])

    pred = model.predict(features).reshape(1, -1)

    return render_template('index.html', strength=pred[0])


# <!--        DRYBULBTEMPF, WETBULBTEMPF, DewPointTempF, RelativeHumidity, WindSpeed, WindDirection, Precip,-->
# <!--                  StationPressure, SeaLevelPressure-->

@app.route('/visibility')
def visibility():
    return render_template("visibility.html")

@app.route("/visibility-check", methods=['POST'])
def predict_visibility():
    DRYBULBTEMPF = int(request.form['DRYBULBTEMPF'])
    WETBULBTEMPF = int(request.form['WETBULBTEMPF'])
    DewPointTempF = int(request.form['DewPointTempF'])
    RelativeHumidity = int(request.form['RelativeHumidity'])
    WindSpeed = int(request.form['WindSpeed'])
    WindDirection = int(request.form['WindDirection'])
    Precip = float(request.form['Precip'])
    StationPressure = float(request.form['StationPressure'])
    SeaLevelPressure = float(request.form['SeaLevelPressure'])
    features = np.array([[DRYBULBTEMPF, WETBULBTEMPF, DewPointTempF, RelativeHumidity, WindSpeed, WindDirection, Precip, StationPressure,SeaLevelPressure]])

    pred = vimodel.predict(features).reshape(1, -1)

    return render_template('visibility.html', strength=pred[0])

if __name__ == "__main__":
    app.run(debug=True)
