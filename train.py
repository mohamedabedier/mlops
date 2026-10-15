import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_real_data():
    """
    Function 1: Load the real California data ONLY when we need it.
    """
    # load the housing data
    housing = fetch_california_housing()
     # ths inputs that tha model train on it like avg bedrooms, house age, avg incom,.......
    x = pd.DataFrame(housing.data, columns=housing.feature_names)
    # this is the outputs that the model predict after training wich is the house price

    y = housing.target
    return x, y


def split_my_data(x , y):

    # we will split the data into 2 groups 80% to train the model 
    # and 20% to test the model prediction accurecy

    # so why not give the model the hole data 100% so he can train better?

    # imagen the model got all the data so when we let the him predict
    # will get all the prediction right but not bec, he trained well
    # it is bec, he train on the all data so he knows what is the prices of all houses
    # so it is not prediction anymore 
    # it is like retriveing the data from a database  

    # so we need to give him only 80% to train and after training give him the other
    # 20% of data witout the prices of the house and lit him predict
    # then compare the real values and the prediction values
    # then calulate the model accurecy

    # ---------------------------------------------------------------------------


    # what is random_state? (it like random_seed when we generat data from python file)
    # we said that we will give the model 80% of the data and this will be random
    # so if we run the code multiable times the 80% of the data will change
    # run 1 got row 1 to 80 
    # run 2 got row 20 to 100 
    # run 3 got row 10 to 90 (for example)
    # so every run train on deffrent data  
    # and we make muliaple runs to see which run is the best so we will move it to production
    # if every run got diffrent data so this is not fare (نو فير نو فير الحكم نو فير)

    # that is why we use random_sate to make every run runs on the same 80% of data


    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=7)

    # but why 42 is the most puopular number? 
    # in a novel called "The Hitchhiker's Guide to the Galaxy"
    # a programmers made a supper cumputer
    # and asked the cumputer "what is the meaning of the life and every thing"
    # and the computer process for 7.5 milion year and then the answer was.......
    # "42" 
    # from that story programers take that number as a random value (دماغهم لسعت والله)

    # I used 7 bec, I am mentally stable (until now)
    
    return X_train, X_val, y_train, y_val


def train_and_evaluate(X_train, X_val, y_train, y_val, max_depth, learning_rate):
    """
    Function 2: Takes data and hyperparameters, trains the model, and returns metrics.
    Isolated so we can test model performance with dummy data if needed.
    """
    # the training of the model
    model = GradientBoostingRegressor(
        max_depth=max_depth, 
        learning_rate=learning_rate, 
        random_state=7
    )
    model.fit(X_train, y_train)
    
    # the model predict and we calculate the metrics
    predictions = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    mae = mean_absolute_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)
    
    return model, rmse, mae, r2


# 2. Define Experiments

    # here we set all the runs to chose only the best one
    # max_depth -> 
    # the depthe of the model image he is asking about every value he gets and is very couis

    # learning_rate -> 
    # is the model learn by fast steps or taking every step slowly to not make a mistake

    # so why not to make the depth 100 (very big) and learning rate 0.00001 (very small)
    # so the model is very couris and deep and takeing every step slowling to learn???
    # ---> overfitting problem


    # also based on what we chose these numbers???

    # 1- basen on known values (Heuristics & Defaults)
    # best depth -> 3-10
    # best learning rate -> 0.01-0.3

    # 2- Grid Search / Random Search
    # in real world we do not write the values maneualy 
    # we give the tool (like GridSearchCV) lists of numbers and it tries 
    # ALL combinations automatically and gives us the best one

    # 3- Smart Search (Bayesian Optimization) -. like Optuna
    # in big companies, we use AI to find the best numbers.
    # if the tool tries depth 50 and gets a bad result, it learns that this is a dead end
    # so it automatically adjusts and tries smaller numbers (like 5 or 6) without human help

EXPERIMENTS_LIST = [
    {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
    {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
    {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
    # let's try that numbers and see the result 
    # {"run_name": "Run 4", "max_depth": 50, "learning_rate": 0.001},
]

def main():
    # 1. Prepare Data 
    X, y = load_real_data()
    X_train, X_val, y_train, y_val = split_my_data(X, y)
    

    # 3. Setup MLflow
    mlflow.set_experiment("House_Price_Prediction")

    # 4. Run and Log
    for exp in EXPERIMENTS_LIST:
        with mlflow.start_run(run_name=exp["run_name"]):
            mlflow.log_param("max_depth", exp["max_depth"])
            mlflow.log_param("learning_rate", exp["learning_rate"])
            
            # Train and evaluate
            model, rmse, mae, r2 = train_and_evaluate(
                X_train, X_val, y_train, y_val, 
                exp["max_depth"], 
                exp["learning_rate"]
            )
            
            # store the metrics
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)
            
            # Log model securely
            mlflow.sklearn.log_model(
                model, 
                "model", 
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )
            
            print(f"{exp['run_name']} Completed -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


if __name__ == '__main__':
    main()