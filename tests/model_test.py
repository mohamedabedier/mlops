import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import load_real_data, split_my_data, train_and_evaluate
from train import EXPERIMENTS_LIST

def test_model_performance_gate():
    """Model CI: Ensure the model meets minimum business requirements"""
    # get the data
    x_test, y_test = load_real_data()
    X_train, X_val, y_train, y_val = split_my_data(x_test, y_test)
    
    # train the model with my best hyperparamiters
    for exp in EXPERIMENTS_LIST:
        model, rmse, mae, r2 = train_and_evaluate(
            x_train, x_val, y_train, y_val, 
            max_depth=exp["max_depth"], 
            learning_rate=exp["learning_rate"]
        )
    
    # check the model quality based on out best run wich is "run 2"
    assert rmse < 0.53, f"Model rejected! RMSE {rmse:.4f} is too high (Must be < 0.53)"
    assert r2 > 0.79, f"Model rejected! R2 score {r2:.4f} is too low (Must be > 0.79)"