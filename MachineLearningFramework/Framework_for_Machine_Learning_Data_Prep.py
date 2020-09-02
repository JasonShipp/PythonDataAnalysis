'''****************************************************************************
Created: on 24/06/2020

Author: Jason Shipp

Purpose: 
    - Defines a function for pre-processing a data frame in preparation for training a model/ making a prediction on an outcome variable
    - Pre-processing steps carried out:
        - Resets the index of the input data
        - Removes columns that should be ignored
        - Converts object columns to numeric if it is possible to do so
        - Removes rows where the outcome variable(s) is missing
        - Removes columns with over 50% missing data
        - Removes rows with over 60% missing data
        - Updates missing values: column mode for non-numeric, column median for numeric/ date/ time delta
        - Replaces numeric outliers by the mean of the column
        - Produces dummy variables from non-numeric columns
        - Transforms date and time delta columns to numeric
        - Standardises numeric values
        - Transforms dense text variables into feature vector columns
        - Normalises numeric values (commented out by default)
****************************************************************************'''

########## Data pre-processing function ##########

def data_preprocessing(is_training, input_data, input_outcome_variables, input_variables_to_ignore, input_dense_text_variables, input_proportion_of_normal_distribution_to_keep):

    working_directory = r"C:/Users/jason/OneDrive/Documents/PythonDataAnalysis/MachineLearningFramework/"

    import joblib
    import pandas as pd
    import re
    import scipy.stats
    import sklearn.feature_extraction  
    
    '''
    - Input data frame is processed to prep it for training/feeding a model
    - Function takes the following inputs:
        - is_training: Bit value. Only used if a dense text variable(s) exists. If 1, a text vectoriser is trained. Otherwise, a pre-trained vectoriser is loaded
        - input_data: Data frame to pre-process
        - input_outcome_variables: List of outcome variable(s) to predict (leave blank if input is live data, i.e. the outcome variables are to be predicted)
        - input_variables_to_ignore: List of variables not to include in model (e.g. row ID columns)
        - input_dense_text_variables: List of variables that contain dense text, to transform into feature vector columns using sklearn.feature_extraction.text.TfidfVectorizer
        - input_proportion_of_normal_distribution_to_keep: Proportion of a normal distribution to treat as non-outlying data
            - Used to calculate the threshold distance from the mean to treat as outlying, to exclude from the training data 
    '''
    
    # Reset input data index
    
    print('Resetting index')
    
    input_data.reset_index(drop = True, inplace = True)

    # Remove columns that should be ignored

    print('Dropping columns marked as to be ignored')
    
    input_data.drop(input_variables_to_ignore, axis = 1, inplace = True)
    
    # Change Object columns to numeric if it is possible to do so
    
    print('Converting object columns to numeric if it is possible to do so')
    
    for col in list(filter(lambda i: not(i in input_outcome_variables), input_data.columns)):
        dtype = input_data[col].dtype.name
        if bool(re.search('object', dtype)):
            if pd.to_numeric(input_data[col].dropna(), errors = 'coerce').notnull().all() == True:
                input_data[col] = pd.to_numeric(input_data[col], errors = 'raise') 

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

    for col in list(filter(lambda i: not(i in (input_outcome_variables + input_dense_text_variables)), input_data.columns)):
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
            if std != 0:
                input_data[col] = input_data[col].apply(lambda x: (x - mean) / std)

    # Transform dense text variables into feature vector columns
    
    for col in input_dense_text_variables:
    
        print('Transforming ' + col + ' dense text variable column into feature vector columns')
        
        if is_training == 1:
            text_vectoriser = sklearn.feature_extraction.text.TfidfVectorizer(
                max_features = 10000
                , decode_error = 'ignore'
                , strip_accents = 'ascii'
                , lowercase = True
                , vocabulary = None
            )
        else:
            text_vectoriser = sklearn.feature_extraction.text.TfidfVectorizer(
                max_features = 10000
                , decode_error = 'ignore'
                , strip_accents = 'ascii'
                , lowercase = True
                , vocabulary = joblib.load(working_directory + col + 'text_vectoriser_vocab.sav')
            )
        
        # Extract text feature vectors (sparse matrix)
        
        text_vectors = text_vectoriser.fit_transform(raw_documents = input_data[col].astype(str))
            
        # Convert text feature vectors into a data frame
        
        text_features_vectorised = pd.DataFrame.sparse.from_spmatrix(
            data = text_vectors
            , columns = ([col + str(i) for i in (pd.RangeIndex(start = 1, stop = text_vectors.shape[1]+1, step = 1).to_list())])
        ).fillna(0)
        # text_features_vectorised.iloc[:, 0].sparse.to_dense().sum() # Check sum of vector values in new variable column
        
        # substitute in new variable columns
        
        input_data = pd.concat(objs = [input_data, text_features_vectorised], axis = 1).drop(col, axis = 1)
        
        # Save text vectoriser if training model
        
        if is_training == 1:
            joblib.dump(text_vectoriser.vocabulary_, working_directory + col + 'text_vectoriser_vocab.sav')        

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
