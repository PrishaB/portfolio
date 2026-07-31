# import pandas

''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path2.
'''
# dataset_2 = pandas.read_csv('nyc_bicycle_counts_2016.csv')
# dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
# dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
# dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
# dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
# dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
# print(dataset_2.to_string()) #This line will print out your data

'''
To do:
1) Find which combination of 3 bridges best predicts total traffic
2) Predict Total using High Temp, Low Temp, and Precipitation
3)Group data by Day and compute the mean number of cyclists per day
  Predict Day of the Week
(finally help visualize this data)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import scikit-learn 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

#use base code from hw10- but regression and not classification
#no need for conf_matrix   -> regression for part 1 and 2
def conf_matrix(y_pred, y_true, num_class):

    num_class = np.unique(y_true) 

    # initialize the confusion matrix to zero
    M = np.zeros((len(num_class), len(num_class)))

    # loop across the different combinations of actual / predicted classes
    for i in range(len(num_class)):
        for j in range(len(num_class)):
            M[i, j] = np.sum((y_true == num_class[i]) & (y_pred == num_class[j]))

    return M

def get_model(name, params):

    if name == "KNN":
        k = params
        model = KNeighborsRegressor(n_neighbors=k) #regressor in place of classifier from hw10?
    else:
        print("ERROR: Model name not recognized/supported. Returned None")
        model = None

    return model

def get_model_results(model_name, params, train_data, train_labels, test_data, test_labels, num_class):
    model = get_model(model_name, params)

    model.fit(train_data, train_labels)
    test_data_new = model.predict(test_data)  
    mse = metrics.mean_squared_error(test_labels, test_data_new)
    # acc = metrics.accuracy_score(test_labels, test_data_new)
    # conf_mat = conf_matrix(test_labels, test_data_new, num_class)
    return mse, model  # Return both MSE and model

if __name__ == "__main__":
    # Cleaning up the csv file
    dataset_2 = pd.read_csv('nyc_bicycle_counts_2016.csv')
    dataset_2['Brooklyn Bridge'] = pd.to_numeric(dataset_2['Brooklyn Bridge'].replace(',', '', regex=True))
    dataset_2['Manhattan Bridge'] = pd.to_numeric(dataset_2['Manhattan Bridge'].replace(',', '', regex=True))
    dataset_2['Queensboro Bridge'] = pd.to_numeric(dataset_2['Queensboro Bridge'].replace(',', '', regex=True))
    dataset_2['Williamsburg Bridge'] = pd.to_numeric(dataset_2['Williamsburg Bridge'].replace(',', '', regex=True))
    dataset_2['Total'] = pd.to_numeric(dataset_2['Total'].replace(',', '', regex=True))
    dataset_2 = dataset_2.dropna()

    # Part 1

       #making combo options for the bridges 
    bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge', 'Williamsburg Bridge']
    combos = [
        ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge'],
        ['Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge'],
        ['Brooklyn Bridge', 'Williamsburg Bridge', 'Queensboro Bridge'],
        ['Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']
    ]
    combo_names = [
        'Brooklyn + Manhattan + Williamsburg',
        'Brooklyn + Manhattan + Queensboro',
        'Brooklyn + Williamsburg + Queensboro',
        'Manhattan + Williamsburg + Queensboro'
    ]

    best_rmse = float('inf')
    best_combo = None

    for i in range(4):   # Iterating through different combos
        X = dataset_2[combos[i]]
        y = dataset_2['Total']

        # Split into train/test //using sklearn package for this here?
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_name = "KNN"
        params = 5
        mse, model = get_model_results(model_name, params, X_train, y_train, X_test, y_test, len(np.unique(y)))

        rmse = np.sqrt(mse)  
        print(f"RMSE for {combo_names[i]}: {rmse:.2f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_combo = combo_names[i]

    print(f"\nBest bridge combination to install sensors: {best_combo}")
    print(f"Lowest RMSE: {best_rmse:.2f}")
    r2 = metrics.r2_score(y_test, model.predict(X_test))
    print(f"R²: {r2:.2f}")

    #  -x-x-x-       -x-x-x-       -x-x-x- 
    # Part 2

#cleaning up the csv 
    dataset_2['High Temp'] = pd.to_numeric(dataset_2['High Temp'])
    dataset_2['Low Temp'] = pd.to_numeric(dataset_2['Low Temp'])
    dataset_2['Precipitation'] = pd.to_numeric(dataset_2['Precipitation'])
# weather features to predict total traffic
    X_forecast = dataset_2[['High Temp', 'Low Temp', 'Precipitation']]
    y_forecast = dataset_2['Total']

# using train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_forecast, y_forecast, test_size=0.2, random_state=42)

# knn regression
    model_name = "KNN"
    params = 5
    mse, model = get_model_results(model_name, params, X_train, y_train, X_test, y_test, len(np.unique(y)))
    rmse = np.sqrt(mse)   

#try computing r^2 metric instead we want ~0.5 (according to piazza) 

    print("\nWeather Forecast Prediction ")

    r2 = metrics.r2_score(y_test, model.predict(X_test))
    print(f"R²: {r2:.2f}")
    print(f"RMSE: {rmse:.2f}")

    #  -x-x-x-       -x-x-x-       -x-x-x- 
    #Part 3

#cleaning up the csv
# from datetime import datetime

dataset_2['Day of Week'] = dataset_2['Day']

#grouping by day 
#and averaging for each bridge
avg_by_day = dataset_2.groupby('Day')[['Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge', 'Williamsburg Bridge']].mean()

print("Average bridge traffic per day:")
print(avg_by_day)


# Prepare features and target
X_day = dataset_2[['Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge', 'Williamsburg Bridge']]
y_day = dataset_2['Day of Week']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_day, y_day, test_size=0.2, random_state=42)

# KNN classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred_day = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred_day)
print(f"\nAccuracy in predicting the day of the week: {accuracy * 100:.2f}%")

conf_matrix = conf_matrix(y_test, y_pred_day, np.unique(y_day))
# print("\nConfusion Matrix for Day Prediction:")
# print(conf_matrix)
#-x-x-x-       -x-x-x-           -x-x-x-

#visualizing the data:

# bridge versus day
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
avg_by_day = avg_by_day.reindex(day_order)    #.reindex??
avg_by_day.plot(kind='line', marker='o')
plt.title("Average Bicycle Traffic by Day of Week")
plt.ylabel("Average Count")
plt.grid(True)
plt.show()

#weather - comparing the number of cyclists during low, high and precipitation 
#(add a trend line maybe for more clear results)
import matplotlib.pyplot as plt
import numpy as np

# Features and titles
weather_features = ['High Temp', 'Low Temp', 'Precipitation']
titles = ['High Temp vs. Total', 'Low Temp vs. Total', 'Precipitation vs. Total']


for i in range(3):
    x = dataset_2[weather_features[i]]
    y = dataset_2['Total']
    
    # trend line
    coeffs = np.polyfit(x, y, 1)     #?? - but it seems to work
    trend = np.poly1d(coeffs)(x)

    # subplot so we can show them 3 side-by-side
    plt.subplot(1, 3, i + 1)
    plt.scatter(x, y, alpha=0.4, label='Data')
    plt.plot(x, trend, color='red', linewidth=2)
    plt.title(titles[i])
    plt.xlabel(weather_features[i])
    if i == 0:
        plt.ylabel("Total Cyclists")
    plt.legend()
    plt.grid(True)

plt.suptitle("Weather vs. Total Bicycle Traffic", fontsize=16)
plt.show()
