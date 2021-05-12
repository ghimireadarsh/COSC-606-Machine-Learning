import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import performance_preprocessor
import pickle

"""
This file provides the performance comparision in the test data

This file first calls pre-processing module for preprocessing the given input
Then uses that pre-preprocessed output for making prediction.

This module compares the output, with the true label as well, to provide the performance comparision between 
prediction of the model and actual label 
"""
import warnings
warnings.filterwarnings("ignore")
print("Succesfully imported modules")

test_df = pd.read_csv("../Dataset/test.csv")
df = test_df.copy(deep=True) # Making the copy of dataset and processing only the copy

print("Pre-processing the data")
df = performance_preprocessor.preprocessor(df)
if df is None:
    print("Pre-processing issue")
else:
    print("Pre-processing completed")
    print("********************************")
    print("Moving forward with prediction")

# Ensembling all the models
def ensemble(X):
    rf_pred = rf.predict(X)
    nn_pred = nn.predict(X)
    gb_pred = gb.predict(X)

    # Weights for different models
    rmse_sums = 7.624 + 12.8071 + 9.8117
    rf_wt = rmse_sums / 7.624
    nn_wt = rmse_sums / 12.8071
    gb_wt = rmse_sums / 9.8117
    rf_value = rf_wt / (rf_wt + nn_wt + gb_wt)
    nn_value = nn_wt / (rf_wt + nn_wt + gb_wt)
    gb_value = gb_wt / (rf_wt + nn_wt + gb_wt)

    ensemble_pred_weighted = rf_pred * rf_value + nn_pred * nn_value + gb_pred * gb_value
    return ensemble_pred_weighted

def regression_results(y_true, y_pred):
    # Regression metrics
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print("R2:", round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('RMSE: ', round(np.sqrt(mse), 4))

if df is not None:
    print("Loading the pre-trained models")
    # Loading the best models
    rf_model_path = "../Models/random_forest.pkl"
    nn_model_path = "../Models/neural_network.pkl"
    gb_model_path = "../Models/boosting_tree.pkl"

    rf = pickle.load(open(rf_model_path, 'rb'))
    nn = pickle.load(open(nn_model_path, 'rb'))
    gb = pickle.load(open(gb_model_path, 'rb'))

    # Dividing the feature column and true label
    print("Successfully loaded pre-trained models")

    selected_features = list(df.columns)
    selected_features.remove("Min Delay")
    target = "Min Delay"
    X = df[selected_features]
    y = df[target]
    print("Making prediction on the test data")
    y_pred = ensemble(X)
    print("Successfully made predictions")
    print("**********************************")
    print("Performance metrics on the data : ")
    regression_results(y, y_pred)

