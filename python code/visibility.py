import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

mydata = pd.read_csv("C:\\Users\\user\\Desktop\\4th sem\\marge_file.csv")
print(mydata.head())

mean_value1 = mydata['WETBULBTEMPF'].mean()
mean_value2 = mydata['DewPointTempF'].mean()

mydata['WETBULBTEMPF'].fillna(value=mean_value1, inplace=True)
mydata['DewPointTempF'].fillna(value=mean_value2, inplace=True)

X = mydata[['DRYBULBTEMPF', 'WETBULBTEMPF', 'DewPointTempF', 'RelativeHumidity', 'WindSpeed', 'WindDirection', 'Precip',
            'StationPressure', 'SeaLevelPressure']]
y = mydata['VISIBILITY']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)  # n_estimators = tree nos.
visibility_data = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
# plt.scatter(y_test, y_pred)
# plt.xlabel('True Visibility')
# plt.ylabel('Predicted Visibility')
# plt.title('True vs. Predicted Visibility')
# plt.show()

import pickle
pickle.dump(visibility_data, open('Visibility.pkl', 'wb'))


def pred_strength(DRYBULBTEMPF, WETBULBTEMPF, DewPointTempF, RelativeHumidity, WindSpeed, WindDirection, Precip,
                  StationPressure, SeaLevelPressure):
    features = np.array([[DRYBULBTEMPF, WETBULBTEMPF, DewPointTempF, RelativeHumidity, WindSpeed, WindDirection, Precip,
                          StationPressure, SeaLevelPressure]])

    pred = visibility_data.predict(features).reshape(1, -1)
    return pred[0]


DRYBULBTEMPF = 68
WETBULBTEMPF = 66.0
DewPointTempF = 64.0
RelativeHumidity = 87
WindSpeed = 13
WindDirection = 120
Precip = 0.0
StationPressure = 29.81
SeaLevelPressure = 29.83
strength = pred_strength(DRYBULBTEMPF, WETBULBTEMPF, DewPointTempF, RelativeHumidity, WindSpeed, WindDirection, Precip,
                         StationPressure, SeaLevelPressure)
print(strength)


#