from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")


@app.route('/')
def login():
    return render_template("login.html")


@app.route('/home')
def home():
    return render_template("home.html")


@app.route('/about_diabetes')
def about_diabetes():
    return render_template("about_diabetes.html")


@app.route('/working')
def working():
    return render_template("working.html")


@app.route('/OurTeam')
def OurTeam():
    return render_template("OurTeam.html")


@app.route('/userinput')
def input():
    return render_template("userinput.html")


@app.route('/userinput', methods=['POST'])
def predict():
    if request.method == 'POST':
        pregnancies = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        blood_pressure = int(request.form['blood_pressure'])
        skin_thickness = int(request.form['skin_thickness'])
        insulin = int(request.form['insulin'])
        BMI = float(request.form['BMI'])
        DPF = float(request.form['DPF'])
        age = int(request.form['age'])
        data = np.array([[pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, BMI, DPF, age]])

        answer = model.predict(data)

        if(answer == 0):
            return render_template("congratulations.html")
        
        else:
            return render_template("alert.html")
app.run(debug = True)
