'''****************************************************************************
Created: on 24/06/2020

Author: Jason Shipp

Purpose: 
    - Imports models generated by the Framework_for_Machine_Learning_Train_Model.py script, pre-processes input data and runs predictions using the imported models
    - The input data structure must align with the data used to train the imported models (run Framework_for_Machine_Learning_Train_Model.py first)
****************************************************************************'''

# Set up parameters

working_directory = r"C:/Users/jason/OneDrive/Documents/PythonDataAnalysis/MachineLearningFramework/" # Directory of Machine Learning scripts

variables_to_ignore = [] # List of variables not to include in model (e.g. row ID columns)

dense_text_variables = [] # List of variables that contain dense text, to transform into feature vector columns using sklearn.feature_extraction.text.TfidfVectorizer

# Import modules

import sys
sys.path.append(working_directory) # Make other Machine Learning scripts accessible

import h2o
import joblib
import numpy as np
import pandas as pd

from Framework_for_Machine_Learning_Data_Prep import data_preprocessing

########## Import data ##########

print('########## Importing data ##########')

outcome_variables = ['species'] # Only needed for test. Otherwise, this should not be in the prediction data

from sklearn.datasets import load_iris
iris = load_iris()

imported = pd.DataFrame(data = iris.data, columns = iris.feature_names)
imported[outcome_variables[0]] = iris.target
imported[outcome_variables[0]]= imported[outcome_variables[0]].map(dict(enumerate(iris.target_names, 0)))

imported_to_predict = imported.copy(deep = True).drop(outcome_variables, axis = 1).sample(random_state = 0, n = int(len(imported)/10))
to_predict = imported_to_predict.copy(deep = True)

########## Load models ##########

print('########## Loading models ##########')

label_encoder = joblib.load(working_directory + 'label_encoder.sav')
important_feature_columns = joblib.load(working_directory + 'important_feature_columns.sav')

model1 = joblib.load(working_directory + 'model1.sav')

h2o.init()
model2 = h2o.load_model(working_directory + 'model2.sav')

########## Process live data to make a prediction on ##########

print('########## Processing live data to make a prediction on ##########')

# Clean and standardise data    

to_predict = data_preprocessing(
    is_training = 0
    , input_data = to_predict
    , input_max_allowed_column_proportion_empty = 1
    , input_max_allowed_row_proportion_empty = 1
    , input_outcome_variables = []
    , input_variables_to_ignore = variables_to_ignore
    , input_dense_text_variables = dense_text_variables
    , input_proportion_of_normal_distribution_to_keep = 1
)

########## Predict outcomes on live data ##########

print('########## Predicting outcomes on live data ##########')

# Subset for important features- populate column with 0s if column is not available in live data (i.e. column is in training dataset only)
to_predict_important_features = pd.DataFrame(index = to_predict.index)

for feature in important_feature_columns:
    if feature in to_predict.columns.tolist():
        to_predict_important_features[feature] = to_predict[feature]
    else:
        to_predict_important_features[feature] = 0
        
# Run predictions

model1_predict_live = model1.predict(X = to_predict_important_features)
imported_to_predict['Model1_Prediction_Raw'] = model1_predict_live

if hasattr(label_encoder, 'classes_'):
    imported_to_predict['Model1_Prediction'] = label_encoder.inverse_transform(np.round(model1_predict_live, 0).astype(int))
else:
    imported_to_predict['Model1_Prediction'] = np.round(model1_predict_live, 0).astype(int)
    
model2_predict_live = model2.predict(test_data = h2o.H2OFrame(to_predict_important_features))
imported_to_predict['Model2_Prediction_Raw'] = model2_predict_live.as_data_frame().iloc[:,0].values

if hasattr(label_encoder, 'classes_'):
    imported_to_predict['Model2_Prediction'] = label_encoder.inverse_transform((model2_predict_live.as_data_frame().round(0).astype(int)).iloc[:,0])
else:
    imported_to_predict['Model2_Prediction'] = model2_predict_live.as_data_frame().round(0).astype(int).iloc[:,0]

print('Predicted outcomes on live data')
print(imported_to_predict)
