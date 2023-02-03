#!/Users/paulhan/repos/kaggle/sp500/venv/bin/python

"""
Project Name: SP500 Model
Description : This application will output the closing price of SP500 given an input date
Author      : Paul (Ki Tae) Han
License     : MIT license
""" 

# import modules
import sqlite3
import pdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# functions
def main():
    # file path
    file_path = './data/sp500_data.sp500_data.csv'

    home_data = pd.read_csv(file_path)

    # Create target object and call it y
    y = home_data.SalePrice

    # Create X
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = home_data[features]

    # Split into validation and training data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    """
    # Specify Model (Decision Tree Regressor)
    iowa_model = DecisionTreeRegressor(random_state=1)

    # Fit Model
    iowa_model.fit(train_X, train_y)
    """

    # Specify Model (Random Forest Regressor)
    iowa_model = RandomForestRegoressor(random_state=1)

    # Fit Model
    iowa_model.fit(train_X, train_y)

    # Make validation predictions and calculate mean absolute error
    val_predictions = iowa_model.predict(val_X)

    # calculate the MAE
    val_mae = mean_absolute_error(val_predictions, val_y)
    print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))


if __name__ == '__main__':
    main()
