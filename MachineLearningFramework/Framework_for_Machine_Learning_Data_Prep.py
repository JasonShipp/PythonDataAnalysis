'''****************************************************************************
Created: on 24/06/2020

Author: Jason Shipp

Purpose: 
    - Defines a function for pre-processing a data frame in preparation for training a model/ making a prediction on an outcome
****************************************************************************'''

########## Data pre-processing function ##########

def data_preprocessing(input_data, input_outcome_variables, input_variables_to_ignore, input_proportion_of_normal_distribution_to_keep):

    import pandas as pd
    import re
    import scipy.stats
    
    '''
    - Input data frame is processed to prep it for training/feeding a model
    - Function takes the following inputs:
        - input_data: Data frame to pre-process
        - input_outcome_variables: List of outcome variable(s) to predict (leave blank if input is live data, i.e. the outcome variables are to be predicted)
        - input_variables_to_ignore: List of variables not to include in model (e.g. row ID columns)
        - input_proportion_of_normal_distribution_to_keep: Proportion of a normal distribution to treat as non-outlying data
            - Used to calculate the threshold distance from the mean to treat as outlying, to exclude from the training data 
    '''

    # Remove columns that should be ignored

    print('Dropping columns marked as to be ignored')
    input_data.drop(input_variables_to_ignore, axis = 1, inplace = True)

    # Remove rows where the outcome variable(s) is missing

    print('Dropping rows where the outcome variable(s) is missing: ' + ', '.join(input_outcome_variables))
    for col in list(filter(lambda i: not(i in input_outcome_variables), input_data.columns)):
        input_data = input_data[input_data[col].apply(lambda x: not(x == '' or pd.isna(x)))]

    # Remove columns with over 50% missing data

    for col in list(filter(lambda i: not(i in input_outcome_variables), input_data.columns)):
        if ((input_data[col].apply(lambda x: 1 if x in [''] or pd.isna(x) else 0).sum())/len(input_data[col])) >= 0.5:
            print('Dropping column ' + col + ' due to too much missing data')
            input_data.drop([col], axis = 1, inplace = True)

    # Remove rows with over 60% missing data

    input_data_col_count = len(input_data.columns) - len(input_outcome_variables)

    for index_val in input_data.index:
        if ((input_data.loc[index_val].apply(lambda x: 1 if x == '' or pd.isna(x) else 0).sum())/input_data_col_count) >= 0.6:
            print('Dropping row ' + str(index_val) + ' due to too much missing data')
            input_data.drop([index_val], axis = 0, inplace = True)         
        
    # Update missing values: column mode for non-numeric, column median for numeric/ date/ time delta

    print('Imputing missing values')
    for col in list(filter(lambda i: not(i in input_outcome_variables), input_data.columns)):
        dtype = input_data[col].dtype.name
        if bool(re.search('object', dtype)) or bool(re.search('category', dtype)):
            input_data[col] = input_data[col].apply(lambda x: input_data[col].mode()[0] if x == '' or pd.isna(x) else x)
        elif bool(re.search('datetime', dtype)) or bool(re.search('timedelta', dtype)):
            input_data[col] = input_data[col].apply(lambda x: (list(input_data.sort_values(col)[col]))[len(input_data)//2] if x == '' or pd.isna(x) else x)
        else:
            input_data[col] = input_data[col].apply(lambda x: input_data[col].median() if x == '' or pd.isna(x) else x)
            
    # Replace numeric outliers by the mean of the column (keeps more information about the outliers than using the median)

    norm_dist_prop_to_keep_stds = scipy.stats.norm.interval(input_proportion_of_normal_distribution_to_keep)[1] # Standard deviations from the mean that capture x proportion of the data, assuming a normal distribution

    print('Replacing numeric values more than ' + str(round(norm_dist_prop_to_keep_stds, 2)) + ' standard deviations from the mean with the column mean')

    for col in list(filter(lambda i: not(i in input_outcome_variables), input_data.columns)):
        dtype = input_data[col].dtype.name
        if not(bool(re.search('object', dtype)) or bool(re.search('category', dtype)) or bool(re.search('datetime', dtype)) or bool(re.search('timedelta', dtype))):
            mean = input_data[col].mean()
            std = input_data[col].std()
            input_data[col] = input_data[col].apply(lambda x: mean if x < (mean - (norm_dist_prop_to_keep_stds * std)) or x > (mean + (norm_dist_prop_to_keep_stds * std)) else x)

    # Produce dummy variables from non-numeric columns

    for col in list(filter(lambda i: not(i in input_outcome_variables), input_data.columns)):
        dtype = input_data[col].dtype.name
        if bool(re.search('object', dtype)) or bool(re.search('category', dtype)):
            print('Adding dummy variable columns to replace ' + col + ' column')
            for elem in input_data[col].unique():
                input_data[str(col + '_' + (re.sub(r'[^\x00-\x7F]+','-', elem)))] = (input_data[col] == elem)
            input_data.drop([col], axis = 1, inplace = True)
            
    # Transform date and time delta columns to numeric

    print('Transforming date and time delta columns to numeric')

    for col in list(filter(lambda i: not(i in input_outcome_variables), input_data.columns)):
        dtype = input_data[col].dtype.name
        if bool(re.search('datetime', dtype)):
            input_data[col] = input_data[col].apply(lambda x: time.mktime(x.timetuple()))
        if bool(re.search('timedelta', dtype)):
            input_data[col] = input_data[col].dt.total_seconds()

    # Standardise numeric values

    print('Rescaling the distribution of numeric values so the mean is 0 and the standard deviation is 1')
    for col in list(filter(lambda i: not(i in input_outcome_variables), input_data.columns)):
        dtype = input_data[col].dtype.name
        if bool(re.search('int', dtype)) or bool(re.search('float', dtype)):
            mean = input_data[col].mean()
            std = input_data[col].std()
            input_data[col] = input_data[col].apply(lambda x: (x - mean) / std)

    '''        
    # Normalise numeric values

    print('Normalising numeric values between 0 and 1')
    for col in list(filter(lambda i: not(i in input_outcome_variables), input_data.columns)):
        dtype = input_data[col].dtype.name
        if bool(re.search('int', dtype)) or bool(re.search('float', dtype)):
            min_val = input_data[col].min()
            max_val = input_data[col].max()
            range_val = max_val - min_val 
            input_data[col] = input_data[col].apply(lambda x: (x - min_val) / range_val)
    '''
    
    return input_data
