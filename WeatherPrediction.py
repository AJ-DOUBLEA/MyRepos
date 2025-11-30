import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random

weather_map = {'Sunny': 1, 'Cloudy': 2, 'Rainy': 3}
reverse_weather_map = {1: 'Sunny', 2: 'Cloudy', 3: 'Rainy'}

temp_data = np.random.randint(20, 35, size=14)
humidity_data = np.random.randint(40, 90, size=14)
weather_data = np.array([random.choice(['Sunny', 'Rainy', 'Cloudy']) for _ in range(14)])
weather_numeric = np.array([weather_map[w] for w in weather_data])

x_days = np.arange(1, 15).reshape(-1, 1)

x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(x_days, temp_data, test_size=0.3, random_state=42)
x_train_h, x_test_h, y_train_h, y_test_h = train_test_split(x_days, humidity_data, test_size=0.3, random_state=42)
x_train_w, x_test_w, y_train_w, y_test_w = train_test_split(x_days, weather_numeric, test_size=0.3, random_state=42)

model_temp = LinearRegression()
model_humidity = LinearRegression()
model_weather = LinearRegression()

model_temp.fit(x_train_t, y_train_t)
model_humidity.fit(x_train_h, y_train_h)
model_weather.fit(x_train_w, y_train_w)

season = input("Enter the season (Dry/Wet): ")
location = input("Enter location: ")

last_3_temp = np.random.randint(20, 35, size=3)
last_3_humidity = np.random.randint(40, 90, size=3)
last_3_weather = np.array([random.choice(['Sunny', 'Rainy', 'Cloudy']) for _ in range(3)])

print("\nLast 3 Days Data:")
print("Temperature:", last_3_temp)
print("Humidity:", last_3_humidity)
print("Weather:", last_3_weather)

next_day_x = np.array([[15]])

pred_temp = model_temp.predict(next_day_x)[0]
pred_humidity = model_humidity.predict(next_day_x)[0]
pred_weather_num = model_weather.predict(next_day_x)[0]
pred_weather = reverse_weather_map[int(round(pred_weather_num))]

print("\nPredictions for tomorrow:")
print("Temperature:", round(pred_temp, 2), "°C")
print("Humidity:", round(pred_humidity, 2), "%")
print("Weather:", pred_weather)

data = pd.DataFrame({
    'Day': np.arange(1, 15),
    'Temperature': temp_data,
    'Humidity': humidity_data,
    'Weather': weather_data
})

print(f"\n{data}")

plt.figure()
plt.title("Temperature (Last 14 Days) with Prediction")
plt.plot(temp_data, marker='o', label='Actual')
plt.plot(x_days, model_temp.predict(x_days), color='red', label='Prediction')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.title("Humidity (Last 14 Days) with Prediction")
plt.plot(humidity_data, marker='o', color='orange', label='Actual')
plt.plot(x_days, model_humidity.predict(x_days), color='red', label='Prediction')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.title("Weather (Last 14 Days) with Prediction")
plt.scatter(range(14), weather_numeric, marker='o', color='green', label='Actual')
plt.plot(x_days, model_weather.predict(x_days), color='red', label='Prediction')
plt.yticks([1, 2, 3], ['Sunny', 'Cloudy', 'Rainy'])
plt.legend()
plt.grid()
plt.show()



