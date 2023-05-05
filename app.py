from flask import Flask, render_template, request

# Serialization refers to the process of converting a data object (e.g., Python objects, Tensorflow models) into a format that allows us to 
# store or transmit the data and then recreate the object when needed using the reverse process of deserialization.

# 'joblib' is a library for performing efficient serialization and deserialization of Python objects.
import joblib
import numpy as np

app = Flask(__name__)

# joblib.load() is used to deserialize the object from the file.
model = joblib.load("model.pkl")

# This line creates a route decorator using the @app.route() syntax, which maps the root URL (/) to the login() function. 
# When a user visits the root URL, Flask calls the login() function, which renders a HTML template called welcome.html using the render_template() function.
# The route() decorator in Flask is used to bind URL to a function
@app.route('/')
def login():
    return render_template("welcome.html")


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