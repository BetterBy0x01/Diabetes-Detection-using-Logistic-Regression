import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('/home/betterby0x01/Diabetes-Detection/diabetes.csv')

X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)

sc = RobustScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model= LogisticRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

joblib.dump(model,"model.pkl")