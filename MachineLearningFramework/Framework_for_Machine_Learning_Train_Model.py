'''****************************************************************************
Created: on 20/06/2020

Author: Jason Shipp

Purpose: 
    - Python framework for data modeling using machine learning, including data preparation, training and testing
    - Script is currently configured to use Iris flower data to train and test Random Forest classification models, using sklearn and H2O libraries
    - The input data and model type are customisable by tweaking the parameters and model-training sections 
    - Models, the parameter label encoder and the important data features are saved in the directory pointed to by the working_directory parameter
    - Data pre-processing is carried out by calling the data_preprocessing function in the Framework_for_Machine_Learning_Data_Prep.py script
    - The above function exports a vocabulary dictionary if text feature vectors are included in the data pre-processing
****************************************************************************'''

# Set up parameters

working_directory = r"C:/Users/jason/OneDrive/Documents/PythonDataAnalysis/MachineLearningFramework/" # Directory of Machine Learning scripts

outcome_variables = ['species'] # List of outcome variable(s) to predict

variables_to_ignore = [] # List of variables not to include in model (e.g. row ID columns)

dense_text_variables = [] # List of variables that contain dense text, to transform into feature vector columns using sklearn.feature_extraction.text.TfidfVectorizer

proportion_of_normal_distribution_to_keep = 0.99 # Proportion of a normal distribution to treat as non-outlying data
# Used to calculate the threshold distance from the mean to treat as outlying, to exclude from the training data 

outcome_variable_values_to_upsample = [] # List of rare values in outcome variable(s) to up-sample in training data
# If using multiple outcome variables, concatenate rare values from multiple columns, separated by '-' (e.g. 'homeowner-male')

threshold_variable_importance = 0.001 # Threshold importance (out of 1) above which to include variables in model. Lower to keep more variables in model

# Import modules

import sys
sys.path.append(working_directory) # Make other Machine Learning scripts accessible

import h2o
import joblib
import numpy as np
import pandas as pd
import re
import shutil
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.model_selection

from Framework_for_Machine_Learning_Data_Prep import data_preprocessing

########## Import data ##########

# Import and summarise data to be used to train and test model (Iris data used as an example)

print('########## Importing data ##########')

from sklearn.datasets import load_iris
iris = load_iris()

imported = pd.DataFrame(data = iris.data, columns = iris.feature_names)
imported['species'] = iris.target
imported['species']= imported['species'].map(dict(enumerate(iris.target_names, 0)))

print('Imported data summary:')
print(imported.describe())

# Load data for training and testing model

raw = imported.copy()    

########## Model data pre-processing ##########
    
print('########## Model data pre-processing ##########')

# Clean and standardise data

raw = data_preprocessing(
    is_training = 1
    , input_data = raw
    , input_outcome_variables = outcome_variables
    , input_variables_to_ignore = variables_to_ignore
    , input_dense_text_variables = dense_text_variables
    , input_proportion_of_normal_distribution_to_keep = proportion_of_normal_distribution_to_keep
)

# Combine outcome variables into 1 column

print('Combining multiple outcome variables into 1 column')

if len(outcome_variables) > 1:
    raw[str('-'.join(outcome_variables))] = raw[outcome_variables].astype(str).agg('-'.join, axis = 1)
    raw.drop(outcome_variables, axis = 1, inplace = True)
    outcome_variables = ['-'.join(outcome_variables)]
        
# Split the data into training and testing data sets

print('Splitting data into training and testing')

train_data, test_data, = sklearn.model_selection.train_test_split(
    raw
    , train_size = 0.8 # % of data to use for training (rest used for testing)
    , random_state = 1
    , shuffle = True
)
        
# Up-sample rare outcome variable in training data to match proportion made up by other variable values

print('Training data before up-sampling outcome variable:')

print(pd.pivot_table(
    data = train_data
    , index = outcome_variables
    , values = train_data.columns[0]
    , aggfunc = 'count'
).rename(columns = {train_data.columns[0]:'Count'})) # Before up-sampling

train_data_variables_no_upsampling = train_data[train_data[outcome_variables[0]].apply(lambda x: not(x in outcome_variable_values_to_upsample))]
train_data_variables_upsampling = train_data[train_data[outcome_variables[0]].apply(lambda x: x in outcome_variable_values_to_upsample)]

if len(outcome_variable_values_to_upsample) > 0:
    train_data_variables_upsampled = train_data_variables_upsampling.sample(n = len(train_data_variables_no_upsampling), replace = True)
    train_data = pd.concat([train_data_variables_no_upsampling, train_data_variables_upsampled], axis = 0)

print('Training data after up-sampling outcome variable:')

print(pd.pivot_table(
    data = train_data
    , index = outcome_variables
    , values = train_data.columns[0]
    , aggfunc = 'count'
).rename(columns = {train_data.columns[0]:'Count'})) # After up-sampling

# Encode the outcome variable(s) as numeric values if not already numeric

label_encoder = sklearn.preprocessing.LabelEncoder()

for col in outcome_variables:
    dtype = train_data[col].dtype.name
    if bool(re.search('object', dtype)) or bool(re.search('category', dtype)):
        print('Encoding outcome variable(s) to numeric: ', col)
        train_data[col] = label_encoder.fit_transform(train_data[col]) # Inverse: label_encoder.inverse_transform(raw[col])
        test_data[col] = label_encoder.fit_transform(test_data[col])     
    
########## Determine important variables using sklearn ##########

print('########## Determining important variables using sklearn ##########')
    
# Pre-fit a Random Forest model, to determine important variables for predicting the outcome variable(s)

model0 = sklearn.ensemble.RandomForestRegressor(random_state = 2, max_depth = 10)

model0.fit(
    X = train_data.drop(outcome_variables, axis = 1)
    , y = train_data[outcome_variables].values.ravel()
)

# Output variable importance

print('Pre-fit model variable importance')

print(pd.DataFrame([
    pd.Series(list(filter(lambda i: not(i in outcome_variables), raw.columns)), name = 'Variable')
    , pd.Series(model0.feature_importances_, name = 'Importance')
]).transpose().sort_values(by = 'Importance', ascending = False))

# Chose important variables to include in model

important_features_model0 = sklearn.feature_selection.SelectFromModel(estimator = model0, prefit = True, threshold = threshold_variable_importance) # Lower threshold to keep more variables in model

important_feature_columns = (np.asarray(list(filter(lambda i: not(i in outcome_variables), raw.columns)))[important_features_model0.get_support()]).tolist()

print('Identified important variables: ' + ', '.join(important_feature_columns))

########## Model using sklearn ##########

print('########## Modeling using sklearn ##########')

# Fit a Random Forest model using only the important variables

model1 = sklearn.ensemble.RandomForestRegressor(random_state = 3, max_depth = 10)

model1.fit(
    X = train_data.drop(outcome_variables, axis=1)[important_feature_columns]
    , y = train_data[outcome_variables].values.ravel()
)

print('Model1 variable importance')

print(pd.DataFrame([
    pd.Series(important_feature_columns, name = 'Variable')
    , pd.Series(model1.feature_importances_, name = 'Importance')
]).transpose().sort_values(by = 'Importance', ascending = False))

# Test model on testing data

model1_predict = model1.predict(X = test_data.drop(outcome_variables, axis = 1)[important_feature_columns])

# Output confusion matrix

print('model1 confusion matrix on test data prediction')

if hasattr(label_encoder, 'classes_'):
    y_pred1 = pd.Series(label_encoder.inverse_transform(np.round(model1_predict, 0).astype(int)), name = 'Predicted').reset_index(drop = True)
    y_actual1 = pd.Series(label_encoder.inverse_transform(test_data[outcome_variables[0]]), name = 'Actual').reset_index(drop = True)
else:
    y_pred1 = pd.Series(np.round(model1_predict, 0).astype(int), name = 'Predicted').reset_index(drop = True)
    y_actual1 = pd.Series(test_data[outcome_variables[0]], name = 'Actual').reset_index(drop = True)

confusion_matrix_model1 = pd.crosstab(y_pred1, y_actual1)

print(confusion_matrix_model1)
print('Model1 successful prediction rate: ' + str(round((np.trace(confusion_matrix_model1)/ np.sum(np.sum(confusion_matrix_model1)))*100, 2)) + '%')

########## Model using H2O ##########

print('########## Modeling using H2O ##########')

# Connect to H2O cluster

h2o.init()

# Fit a Random Forest model using only the important variables

model2 = h2o.estimators.random_forest.H2ORandomForestEstimator(balance_classes = False, binomial_double_trees = True,  max_depth = 10, seed = 0)

model2.train(
    x = important_feature_columns
    , y = outcome_variables[0]
    , training_frame = h2o.H2OFrame(train_data)
)

print('Model2 summary')
print(model2)

# Test model on testing data

pred2 = model2.predict(test_data = h2o.H2OFrame(test_data))

# Output confusion matrix

print('model2 confusion matrix on test data prediction')

if hasattr(label_encoder, 'classes_'):
    y_pred2 = pd.Series(label_encoder.inverse_transform((pred2.as_data_frame().round(0).astype(int))['predict']), name = 'Predicted').reset_index(drop = True)
    y_actual2 = pd.Series(label_encoder.inverse_transform(test_data[outcome_variables[0]]), name = 'Actual').reset_index(drop = True)
else:
    y_pred2 = pd.Series((pred2.as_data_frame().round(0).astype(int))['predict'], name = 'Predicted').reset_index(drop = True)
    y_actual2 = pd.Series(test_data[outcome_variables[0]], name = 'Actual').reset_index(drop = True)
    
confusion_matrix_model2 = pd.crosstab(y_pred2, y_actual2)

print(confusion_matrix_model2)
print('Model2 successful prediction rate: ' + str(round((np.trace(confusion_matrix_model2)/ np.sum(np.sum(confusion_matrix_model2)))*100, 2)) + '%')

########## Save models ##########

print('########## Saving models ##########')

joblib.dump(label_encoder, working_directory + 'label_encoder.sav')
joblib.dump(important_feature_columns, working_directory + 'important_feature_columns.sav')

joblib.dump(model1, working_directory + 'model1.sav')

model2_dir = h2o.save_model(model = model2, path = working_directory, force = True) # Force overwriting
shutil.move(src = model2_dir, dst = working_directory + 'model2.sav')

# Close connection to H2O cluster 

h2o.cluster().shutdown()
