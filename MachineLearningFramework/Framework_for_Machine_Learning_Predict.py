'''****************************************************************************
Created: on 24/06/2020

Author: Jason Shipp

Purpose: 
    - Import models generated by Framework_for_Machine_Learning_Train_Model.py script, pre-process data and run predictions
****************************************************************************'''

# Set up parameters

working_directory = r"C:/Users/jason/OneDrive/Documents/PythonDataAnalysis/MachineLearningFramework/"

dense_text_variables = [] # List of variables that contain dense text, to transform into feature vector columns using sklearn.feature_extraction.text.TfidfVectorizer

# Import modules

import sys
sys.path.append(working_directory) # Directory of custom Machine Learning scripts

import pandas as pd
import numpy as np
import h2o
import joblib
from Framework_for_Machine_Learning_Data_Prep import data_preprocessing

########## Import data ##########

print('########## Importing data ##########')

outcome_variables = ['species']

from sklearn.datasets import load_iris
iris = load_iris()

imported = pd.DataFrame(data = iris.data, columns = iris.feature_names)
imported[outcome_variables[0]] = iris.target
imported[outcome_variables[0]]= imported[outcome_variables[0]].map(dict(enumerate(iris.target_names, 0)))

imported_to_predict = imported.copy().drop(outcome_variables, axis = 1).sample(random_state = 0, n = int(len(imported)/10))
to_predict = imported_to_predict.copy()

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
    , input_outcome_variables = []
    , input_variables_to_ignore = []
    , input_dense_text_variables = dense_text_variables
    , input_proportion_of_normal_distribution_to_keep = 1
)

########## Predict outcomes on live data ##########

print('########## Predicting outcomes on live data ##########')

model1_predict_live = model1.predict(X = to_predict[important_feature_columns])
imported_to_predict['Model1_Prediction'] = label_encoder.inverse_transform(np.round(model1_predict_live, 0).astype(int))

model2_predict_live = model2.predict(test_data = h2o.H2OFrame(to_predict[important_feature_columns]))
imported_to_predict['Model2_Prediction'] = label_encoder.inverse_transform((model2_predict_live.as_data_frame().round(0).astype(int))['predict'])

print('Predicted outcomes on live data')
print(imported_to_predict)

print('Actual outcomes in live data')
print(imported.copy().sample(random_state = 0, n = int(len(imported)/10)))
