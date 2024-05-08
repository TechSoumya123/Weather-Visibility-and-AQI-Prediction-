import pandas as pd

mydata = pd.read_csv("C:\\Users\\user\\Desktop\\4th sem\\Air quality index.csv")

from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
mydata
imputer = imputer.fit(mydata)
mydata = imputer.transform(mydata)

mydata = pd.DataFrame(mydata)

X = mydata.iloc[:, 0:-1].values
Y = mydata.iloc[:, -1].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=30, random_state=30)

mydata_air = model.fit(X_train, Y_train)

print('The training r_sq is: %.2f' % mydata_air.score(X_train, Y_train))

import pickle
pickle.dump(mydata_air, open('RandomModel.pkl', 'wb'))


def pred_strength(T, TM, Tm, SLP, H, VV, V, VM):
    features = np.array([[T, TM, Tm, SLP, H, VV, V, VM]])

    pred = mydata_air.predict(features).reshape(1, -1)
    return pred[0]


T = 7.8
TM = 12.7
Tm = 4.4
SLP = 1018.5
H = 87
VV = 0.6
V = 4.4
VM = 11.1
strength = pred_strength(T, TM, Tm, SLP, H, VV, V, VM)
print(strength)
