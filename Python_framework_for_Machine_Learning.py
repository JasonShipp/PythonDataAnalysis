'''****************************************************************************
Created: on 20/06/2020

Author: Jason Shipp

Purpose: 
    - Python framework for data modeling using machine learning, including data preparation, training, testing and prediction
    - Iris flower data set used as an example to train and test Random Forest classification models, using sklearn and h2o libraries
****************************************************************************'''

# Set up parameters

outcome_variables = ['species'] # Expects a list

variables_to_ignore = [] # Expects a list: variables not to include in model (e.g. row ID columns)

outcome_variable_values_to_upsample = [] # Expects a list: rare values in outcome variable(s) to up-sample in training data
# If using multiple outcome variables, concatenate rare values from multiple columns, separated by '-' (e.g. 'homeowner-male')

proportion_of_normal_distribution_to_keep = 0.99 # Proportion of a normal distribution to treat as non-outlying data. Used to calculate the threshold distance from the mean to treat as outlying 

threshold_variable_importance = 0.001 # Threshold importance (out of 1) above which to include variables in model. Lower to keep more variables in model

# Import modules

import pandas as pd
import numpy as np
import re
import scipy.stats as stats
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.model_selection
import h2o

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

# Load (fake) live data to apply model to

imported_to_predict = imported.copy().drop(outcome_variables, axis = 1).sample(random_state = 0, n = int(len(imported)/10))
to_predict = imported_to_predict.copy()

########## Data pre-processing function ##########

def data_preprocessing(input_data):

    '''
	- Function takes a data frame as an input
	- Data is processed to prep it for training/feeding a model
	- Global parameters control feature/predictor variables and model parameters
	'''

    global outcome_variables
    global variables_to_ignore
    global outcome_variable_values_to_upsample
    global proportion_of_normal_distribution_to_keep 
    global threshold_variable_importance

    # Remove columns that should be ignored

    print('Dropping columns marked as to be ignored')
    input_data.drop(variables_to_ignore, axis = 1, inplace = True)

    # Remove rows where the outcome variable(s) is missing

    print('Dropping rows where the outcome variable(s) is missing: ' + ', '.join(outcome_variables))
    for col in list(filter(lambda i: not(i in outcome_variables), input_data.columns)):
        input_data = input_data[input_data[col].apply(lambda x: not(x == '' or pd.isna(x)))]

    # Remove columns with over 50% missing data

    for col in list(filter(lambda i: not(i in outcome_variables), input_data.columns)):
        if ((input_data[col].apply(lambda x: 1 if x in [''] or pd.isna(x) else 0).sum())/len(input_data[col])) >= 0.5:
            print('Dropping column ' + col + ' due to too much missing data')
            input_data.drop([col], axis = 1, inplace = True)

    # Remove rows with over 60% missing data

    input_data_col_count = len(input_data.columns) - len(outcome_variables)

    for index_val in input_data.index:
        if ((input_data.loc[index_val].apply(lambda x: 1 if x == '' or pd.isna(x) else 0).sum())/input_data_col_count) >= 0.6:
            print('Dropping row ' + str(index_val) + ' due to too much missing data')
            input_data.drop([index_val], axis = 0, inplace = True)         
        
    # Update missing values: column mode for non-numeric, column median for numeric/ date/ time delta

    print('Imputing missing values')
    for col in list(filter(lambda i: not(i in outcome_variables), input_data.columns)):
        dtype = input_data[col].dtype.name
        if bool(re.search('object', dtype)) or bool(re.search('category', dtype)):
            input_data[col] = input_data[col].apply(lambda x: input_data[col].mode()[0] if x == '' or pd.isna(x) else x)
        elif bool(re.search('datetime', dtype)) or bool(re.search('timedelta', dtype)):
            input_data[col] = input_data[col].apply(lambda x: (list(input_data.sort_values(col)[col]))[len(input_data)//2] if x == '' or pd.isna(x) else x)
        else:
            input_data[col] = input_data[col].apply(lambda x: input_data[col].median() if x == '' or pd.isna(x) else x)
            
    # Replace numeric outliers by the mean of the column (keeps more information about the outliers than using the median)

    norm_dist_prop_to_keep_stds = stats.norm.interval(proportion_of_normal_distribution_to_keep)[1] # Standard deviations from the mean that capture proportion_of_normal_distribution_to_keep proportion of the data, assuming a normal distribution

    print('Replacing numeric values more than ' + str(round(norm_dist_prop_to_keep_stds, 2)) + ' standard deviations from the mean with the column mean')

    for col in list(filter(lambda i: not(i in outcome_variables), input_data.columns)):
        dtype = input_data[col].dtype.name
        if not(bool(re.search('object', dtype)) or bool(re.search('category', dtype)) or bool(re.search('datetime', dtype)) or bool(re.search('timedelta', dtype))):
            mean = input_data[col].mean()
            std = input_data[col].std()
            input_data[col] = input_data[col].apply(lambda x: mean if x < (mean - (norm_dist_prop_to_keep_stds * std)) or x > (mean + (norm_dist_prop_to_keep_stds * std)) else x)

    # Produce dummy variables from non-numeric columns

    for col in list(filter(lambda i: not(i in outcome_variables), input_data.columns)):
        dtype = input_data[col].dtype.name
        if bool(re.search('object', dtype)) or bool(re.search('category', dtype)):
            print('Adding dummy variable columns to replace ' + col + ' column')
            for elem in input_data[col].unique():
                input_data[str(col + '_' + (re.sub(r'[^\x00-\x7F]+','-', elem)))] = (input_data[col] == elem)
            input_data.drop([col], axis = 1, inplace = True)
            
    # Transform date and time delta columns to numeric

    print('Transforming date and time delta columns to numeric')

    for col in list(filter(lambda i: not(i in outcome_variables), input_data.columns)):
        dtype = input_data[col].dtype.name
        if bool(re.search('datetime', dtype)):
            input_data[col] = input_data[col].apply(lambda x: time.mktime(x.timetuple()))
        if bool(re.search('timedelta', dtype)):
            input_data[col] = input_data[col].dt.total_seconds()

    # Standardise numeric values

    print('Rescaling the distribution of numeric values so the mean is 0 and the standard deviation is 1')
    for col in list(filter(lambda i: not(i in outcome_variables), input_data.columns)):
        dtype = input_data[col].dtype.name
        if bool(re.search('int', dtype)) or bool(re.search('float', dtype)):
            mean = input_data[col].mean()
            std = input_data[col].std()
            input_data[col] = input_data[col].apply(lambda x: (x - mean) / std)

    '''        
    # Normalise numeric values

    print('Normalising numeric values between 0 and 1')
    for col in list(filter(lambda i: not(i in outcome_variables), input_data.columns)):
        dtype = input_data[col].dtype.name
        if bool(re.search('int', dtype)) or bool(re.search('float', dtype)):
            min_val = input_data[col].min()
            max_val = input_data[col].max()
            range_val = max_val - min_val 
            input_data[col] = input_data[col].apply(lambda x: (x - min_val) / range_val)
    '''
	
    return input_data
	
########## Model data pre-processing ##########
	
print('########## Model data pre-processing ##########')

raw = data_preprocessing(input_data = raw)

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
	, train_size = 0.8 # % of data to use for training (rets used for testing)
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

print('Encoding outcome variable(s) if not already numeric: ' + ', '.join(outcome_variables))
for col in outcome_variables:
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
y_pred1 = pd.Series(label_encoder.inverse_transform(np.round(model1_predict, 0).astype(int)), name = 'Predicted').reset_index(drop = True)
y_actual1 = pd.Series(label_encoder.inverse_transform(test_data[outcome_variables[0]]), name = 'Actual').reset_index(drop = True)
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
y_pred2 = pd.Series(label_encoder.inverse_transform((pred2.as_data_frame().round(0).astype(int))['predict']), name = 'Predicted').reset_index(drop = True)
y_actual2 = pd.Series(label_encoder.inverse_transform(test_data[outcome_variables[0]]), name = 'Actual').reset_index(drop = True)
confusion_matrix_model2 = pd.crosstab(y_pred2, y_actual2)
print(confusion_matrix_model2)
print('Model2 successful prediction rate: ' + str(round((np.trace(confusion_matrix_model2)/ np.sum(np.sum(confusion_matrix_model2)))*100, 2)) + '%')

########## Process live data to make a prediction on ##########

print('########## Processing live data to make a prediction on ##########')

to_predict = data_preprocessing(input_data = to_predict)

########## Predict outcomes on live data ##########

print('########## Predicting outcomes on live data ##########')

model1_predict_live = model1.predict(X = to_predict[important_feature_columns])
imported_to_predict['Model1_Prediction'] = label_encoder.inverse_transform(np.round(model1_predict_live, 0).astype(int))

model2_predict_live = model2.predict(test_data = h2o.H2OFrame(to_predict[important_feature_columns]))
imported_to_predict['Model2_Prediction'] = label_encoder.inverse_transform((model2_predict_live.as_data_frame().round(0).astype(int))['predict'])

print('Predicted outcomes on live data')
print(imported_to_predict)

# Close connection to H2O cluster 

h2o.cluster().shutdown()
