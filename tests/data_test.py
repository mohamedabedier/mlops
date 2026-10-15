import pandas as pd 
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import load_real_data

def test_no_null_values():
    """this func check for missing values and cols and Value bound in the dataset"""
    # get the data
    x_test, y_test = load_real_data()
    
    assert x_test.isnull().sum().sum() == 0, "input data contains NULL values!"
    assert pd.Series(y_test).isnull().sum().sum() == 0, "output data contains NULL values!"

    assert (y_test > 0).all(), "Value bound: House price cannot be zero or negative!"
    assert (x_test['MedInc'] > 0).all(), "Value bound: MedInc cannot be zero or negative!"
    assert (x_test['HouseAge'] > 0).all(), "Value bound: HouseAge cannot be zero or negative!"
    assert (x_test['AveRooms'] > 0).all(), "Value bound: AveRooms cannot be zero or negative!"
    assert (x_test['AveBedrms'] > 0).all(), "Value bound: AveBedrms cannot be zero or negative!"
    assert (x_test['Population'] > 0).all(), "Value bound: Population cannot be zero or negative!"
   

    
    expected_columns = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
        'Population', 'AveOccup', 'Latitude', 'Longitude'
    ]
    
    for col in expected_columns:
        assert col in x_test.columns, f"Missing critical column in data: {col}"