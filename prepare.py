# prepare.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

FILL_NA_FIELDS = ["PCR_02", "household_income"]
SPECIAL_PROPERTY_BLOODTYPES = ["O+", "B+"]
MINMAX_NORMALIZE_FIELDS = ["PCR_01","PCR_03","PCR_04","PCR_06","PCR_08"]
STANDARD_NORMALIZE_FIELDS = ["PCR_02","PCR_05","PCR_07","PCR_09","PCR_10"]

# Helper function, makes a list of medians for required fields
def calc_fields_median(training_data, fields):
  medians = []

  for field in fields:
    medians.append(training_data[field].median())

  return medians

def prepare_data(training_data, new_data):
  # We will work on a copy of new_data to not change it
  new_data_processed = new_data.copy()

  # Calculate required medians from training_data to fill new_data
  medians = calc_fields_median(training_data, FILL_NA_FIELDS)

  # Fill copy of new_data null values with medians calculated before.
  for field, field_median in zip(FILL_NA_FIELDS, medians):
    new_data_processed[field] = new_data_processed[field].fillna(field_median)

  # Calculate Special Property
  new_data_processed['SpecialProperty'] = new_data.copy()["blood_type"].isin(SPECIAL_PROPERTY_BLOODTYPES).astype(int)

  # Initialize scalers
  mm_scaler = MinMaxScaler(feature_range=(-1,1))
  std_scaler = StandardScaler()

  # Apply scalers on required columns
  new_data_processed[MINMAX_NORMALIZE_FIELDS] = mm_scaler.fit_transform(new_data_processed[MINMAX_NORMALIZE_FIELDS])
  new_data_processed[STANDARD_NORMALIZE_FIELDS] = StandardScaler().fit_transform(new_data_processed[STANDARD_NORMALIZE_FIELDS])

  return new_data_processed

