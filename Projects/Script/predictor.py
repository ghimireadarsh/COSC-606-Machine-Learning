import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import predictor_preprocessor
import pickle

"""
This module makes prediction on the data provided.

This file first calls predictor_preprocessor module for pre-processing the given input
Then uses that pre-processed output for making prediction.

This module requires the following inputs:
['Report Date', 'Route', 'Time', 'Day', 'Location', 'Direction', 'Min Gap']

It is not necessary to have proper info in Location feature however the other information should 
be provided accurately for having accurate prediction.

Then, uses this data for making prediction of Min Delay that will occur.
"""
import warnings
warnings.filterwarnings("ignore")
print("Succesfully imported modules")

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

# Loading the data
print("Loading the data...")
# test_df = pd.read_csv("../Dataset/test.csv")
test_df = pd.read_csv("../Dataset/one_data.csv")
df = test_df.copy(deep=True) # Making the copy of dataset and processing only the copy
print("Data loading success !")
print("Pre-processing the data...")
try:
    selected_columns = ['Report Date', 'Route', 'Time', 'Day', 'Location', 'Direction', 'Min Gap']
    X = df[selected_columns]
    X = predictor_preprocessor.preprocessor(X)
    print("Pre-processing completed !")
    print("********************************")
    print("Moving forward with prediction...")

    print("Loading the pre-trained models...")
    # Loading the best models
    rf_model_path = "../Models/random_forest.pkl"
    nn_model_path = "../Models/neural_network.pkl"
    gb_model_path = "../Models/boosting_tree.pkl"

    rf = pickle.load(open(rf_model_path, 'rb'))
    nn = pickle.load(open(nn_model_path, 'rb'))
    gb = pickle.load(open(gb_model_path, 'rb'))

    # Dividing the feature column and true label
    print("Successfully loaded pre-trained models...")

    print("Making prediction on the test data...")
    y_pred = ensemble(X)
    print("Successfully made predictions !")
    print("**********************************")
    print("Minimum delay predicted in minutes is : ")
    print(y_pred)
except:
    print("Data provided has Issue !")
    print("Please provide data with these features : ['Report Date', 'Route', 'Time', 'Day', 'Location', 'Direction', 'Min Gap']")

