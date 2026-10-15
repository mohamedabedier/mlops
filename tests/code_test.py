import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import load_real_data, split_my_data

def test_data_split_ratio():
    """this test check if the split func splits 80/20 and all the fetaure have values"""
    # get the data
    x_test, y_test = load_real_data()
    X_train, X_val, y_train, y_val = split_my_data(x_test, y_test)
    
    total_len = len(X_train) + len(X_val)
    expected_train_len = int(total_len * 0.80)
    
    # check that the training data is 80%
    assert abs(len(X_train) - expected_train_len) <= 1, f"Data split is not 80/20! - we expect {expected_train_len} rows and we got {len(X_train)}"
    
    # check that every col/fetaure/x have value
    assert len(X_train) == len(y_train), f"Mismatch between x_train and y_train lengths - x lenght = {len(X_train)} and y lenght = {len(y_train)}"




    