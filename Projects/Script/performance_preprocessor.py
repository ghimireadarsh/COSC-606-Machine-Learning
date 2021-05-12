import numpy as np
import pandas as pd
import pickle

"""
This file preprocesses data from the dataframe
All the parameters are obtained from data used for training, thus every preprocessing is done 
using the training data parameters

The preprocessor will not process the data unless the data provided feature columns are as per the requirement
"""
import warnings
warnings.filterwarnings("ignore")

def clean_delay(df):
  df = df[df['Min Delay'].notna()]
  return df

def check_route(x):
    # This function checks for valid routes only
    # Valid routes are identified from the website, only routes where the streetcar are working

    # load the valid list of TTC Streetcar routes
  valid_routes = [501, 502, 503, 504, 505, 506, 509, 510, 511, 512, 301, 304, 306, 310]

  if x in valid_routes:
    return x
  else:
    return "bad route"

# This function cleans the data based on valid routes
def clean_route(df):
  # This function takes dataframe as input
  # cleans the route column based on the validity of the route of street car
  # returns the cleaned dataframe
  df['Route'] = df['Route'].apply(lambda x:check_route(x))
  df = df[df.Route != "bad route"]
  df['Route'] = df['Route'].astype('int64')
  return df

# This function drops the Location column
def drop_location(df):
  df = df.drop(["Location"], axis=1)
  return df

def create_date_time_column(df):
   # This function takes dataframe, then merges the date and time
   # Then convert that column into datetime datatype
   # Such that it can be further used in time series easily
  try:
    new = pd.to_datetime(df["Report Date"] + " "+ df["Time"], utc=True)
    df["Date Time"] = new
    df = df.drop(["Report Date", "Time"], axis=1)
    return df
  except:
    return df

# This function divides a day into different period
def day_divider(hour):
  if hour > 5 and hour < 12:
    return "morning"
  elif hour >= 12 and hour < 17:
    return "afternoon"
  elif hour >= 17 and hour < 21:
    return "evening"
  else:
    return "night"


min_gap_scaler_data = pickle.load(open("../Models/min_gap_scaler.pkl", 'rb'))
min_gap_train_mean = min_gap_scaler_data["mean"]
min_gap_train_std = min_gap_scaler_data["std"]

def clean_gap(df):
  # This function will help to clean the Min Gap column feature with training data Min Gap mean value
  df["Min Gap"] = df["Min Gap"].fillna(min_gap_train_mean)
  return df

# These function help to filter the Direction values and clean them

def check_direction (x):
    valid_directions = ['eb', 'wb', 'nb', 'sb', 'bw']
    if x in valid_directions:
        return(x)
    else:
        return("bad direction")

def direction_cleanup(df):
    df['Direction'] = df['Direction'].str.lower()
    df['Direction'] = df['Direction'].str.replace('/','')
    df['Direction'] = df['Direction'].replace({'eastbound':'eb','westbound':'wb','southbound':'sb','northbound':'nb'})
    df['Direction'] = df['Direction'].apply(lambda x:check_direction(x))
    return(df)

def complete_cleaner(df):
  df = clean_delay(df) # drops the nan Min delay rows
  df = clean_route(df) # cleans the unwanted route from the dataset
  df = drop_location(df) # drops the location column from the dataset
  df = create_date_time_column(df) # creates Date Time column in the dataset
  df["Part of Day"] = df.apply(lambda x: day_divider(x["Date Time"].hour),axis=1) # Creates Part of Day column in the dataset
  df = clean_gap(df) # cleans gap based on the mean of gap values
  df = direction_cleanup(df) # cleans the direction column to 5 directions(eb,wb,nb,sb,bw)
  df = df[df["Direction"] != "bad direction"]
  df.reset_index(inplace=True, drop=True)
  df.drop(['Date Time'], axis=1, inplace=True)
  return df

day_enc = pickle.load(open("../Models/day_encoder.pkl", 'rb'))
route_enc = pickle.load(open("../Models/route_encoder.pkl", 'rb'))
dir_enc = pickle.load(open("../Models/direction_encoder.pkl", 'rb'))
part_enc = pickle.load(open("../Models/part_of_day_encoder.pkl", 'rb'))

def one_hot_encoder(df):
  # This function does one hot encoding of the day, route, direction, and part of day columns
  # The one hot encoder object for different features are created using training data
  # The same objects will be used for test data encoding
  df[day_enc.categories_[0]] = day_enc.transform(np.array(df["Day"]).reshape(-1,1)).toarray()
  df[route_enc.categories_[0]] = route_enc.transform(np.array(df["Route"]).reshape(-1,1)).toarray()
  df[dir_enc.categories_[0]] = dir_enc.transform(np.array(df["Direction"]).reshape(-1,1)).toarray()
  df[part_enc.categories_[0]] = part_enc.transform(np.array(df["Part of Day"]).reshape(-1,1)).toarray()

  df.drop(['Day', 'Route', 'Direction', 'Part of Day'], axis=1, inplace=True)
  return df


def min_gap_scaler(df):
  df["Min Gap"] = (df["Min Gap"]-min_gap_train_mean)/min_gap_train_std
  return df

def preprocessor(df):
  # This function does all the preprocessing
  selected_columns = ['Report Date', 'Route', 'Time', 'Day', 'Location', 'Direction',
       'Min Delay', 'Min Gap']
  try:
      df = df[selected_columns]
      df = complete_cleaner(df)
      df = one_hot_encoder(df)  # one hot encoder based on training samples
      df = min_gap_scaler(df)  # feature scaler based on training sample
      print("------------------------")
      if df.isnull().values.any():
          print("There are some null values")
      else:
          print("Data is ready for testing")
      return df
  except:
      print("The columns of the data are not matching and sufficient")
      print("Provide proper data with the following headers : ", selected_columns)




