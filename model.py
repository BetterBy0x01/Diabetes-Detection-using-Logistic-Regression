import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/ronak/Desktop/flask/diabetes.csv')
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)
lr = RandomForestClassifier()
lr.fit(X_train, Y_train)
plt.plot(X, lr.predict(X))

import joblib

joblib.dump(lr,"model.pkl")

m = joblib.load("model.pkl")
# m.predict([[2]])