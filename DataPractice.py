import numpy as np
import pandas as pd 
import random
import matplotlib.pyplot as plt
import csv

filename = 'CityTemperatures.csv'

temperatures = np.random.randint(20, 35, size=(7, 3))   

average_temperature_city_a = np.mean(temperatures[:, 0])
average_temperature_city_b = np.mean(temperatures[:, 1])
average_temperature_city_c = np.mean(temperatures[:, 2])
highest_average = max(average_temperature_city_a, average_temperature_city_b, average_temperature_city_c)

def write_to_csv(filename, temperatures):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Day', 'City A', 'City B', 'City C'])
        
        for i, temp in enumerate(temperatures):
            writer.writerow([f'Day {i+1}'] + temp.tolist())
    
    print(f"\n\nSuccessfully written to {filename}... ")

data = pd.DataFrame(temperatures, columns=['City A', 'City B', 'City C'])
data.index = [f'Day {i+1}' for i in range(7)]


print("\nTemperature DataFrame:")
print(data)

print(f"\n\nAverage Temperature City A: {average_temperature_city_a:.2f}°C")
print(f"Average Temperature City B: {average_temperature_city_b:.2f}°C")
print(f"Average Temperature City C: {average_temperature_city_c:.2f}°C")
print("Highest Average Temperature: ", highest_average)



plt.plot(temperatures, marker='o')
plt.title('Daily Temperatures in Three Cities Over a Week')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.grid()
plt.show()

