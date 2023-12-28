# -*- coding: utf-8 -*-
""" 
Chemical Fingerprinting Workflow Software by Gerrad Jones, 
Anna Beran, Corey De La Cruz, Cheng Shi, Quirine Van Swaay,
and Jane Vinesky.

Copyright 2022 Oregon State University. All rights reserved.

License Notice:

The Chemical Fingerprinting Workflow Software found in this GitHub 
repository (the "Software") may be freely used for educational 
and research purposes by non-profit institutions and United States
Federal Government agencies only. Other organizations are allowed
to use the Software for evaluation purposes only. Any further uses
will require prior approval, and the Software may not be sold or
redistributed without prior approval.  Portions of this Software
are also subject to pending patents, with rights available from
Oregon State University on request to advantage@oregonstate.edu.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE 
OR OTHER DEALINGS IN THE SOFTWARE.

By downloading, compiling or executing any part of this Software 
constitutes your agreement to these terms.
 
---
GitHub Repository: https://github.com/EcoChem-OSU
Software Disclosure References: OSU-19-31; OSU-22-02
"""
import numpy as np
import random
import timeit
import scikitplot as skplt
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class fingerprintParametersClass():
    importance_iterations = None
    final_iterations = None
    num_features = None
    test_size = None
    n_combos = None
    n_rs = None
    
class fingerprintSVCParametersClass():
    c_lower = None
    c_upper = None
    c_divisions = None

class storingParametersClass():
    classifier = None
    results = None

class modeledResultsClass():
        resultsSorted = None
        label = None
        hyperParameterTop = None
        bestParamsTop = None

class importanceMetricsClass():
    impMean = None
    impStd = None
    absMean = None
    importance = None

def selectFolder():
    from tkinter import Tk
    from tkinter import filedialog
    #Use Tk() as root.
    root = Tk() 
    #Hide GUI for Tkinter
    root.withdraw() 
    #Show directory selection.
    root.attributes('-topmost', True) 
    
    open_file = filedialog.askdirectory() # Returns opened path as str
    return open_file


def createFolderPath(sourceDirectory, folderName, verbose = False):
    """Creates a new folder in specified directory with supplied name."""
    import os
    #Confirm supplied directory is valid
    
    pathTest = os.path.isdir(sourceDirectory)
    #If path does not exist, raise exception and exit script.
    sourceDirectory = sourceDirectory
    if(pathTest == False):
        raise ValueError('Error: "'+ str(sourceDirectory) +'" is not a valid location. Please provide valid location and rerun script.')
    else:
        pass
    
    #Checks to see if folder requested to create already exists.
    exists = os.path.isdir(sourceDirectory + "/" + folderName)
    #IF it does not exist, attempt to create folder.
    if(exists == False):
        try:
            os.mkdir(sourceDirectory + "/" + folderName)
            if(verbose == True):
                print("Folder path created: " + sourceDirectory + "/" + folderName)
        #IF a name error occurs, print error
        except OSError as error:
            print(error)
            raise ValueError('Error: Folder name "' + folderName + '" is not a valid name for a folder. Please provide a valid name without invalid characters.')
    #If folder already exists, does not create folder.
    else:
        #If verbose is True, inform user that path already exists.
        if (verbose == True):
            print('"File path: "' + sourceDirectory + "/" + folderName + '" already exists. No new folder created.')
        else:
            pass
        
def testFingerprintFilePaths(sourceDirectory):
    """Tests that required source files exists in provided directory."""
    import os
    #Test file paths in directory.
    xDataTest = os.path.isfile(sourceDirectory +'/X.txt')
    if(xDataTest == False):
        raise ValueError('Error: X dataset does not exist. Expected: "'+sourceDirectory+'/X.txt". Please check source directory.')
    else:
        pass
    yDataTest = os.path.isfile(sourceDirectory + '/y.txt')
    if(yDataTest == False):
        raise ValueError('Error: Y dataset does not exist. Expected: "' + sourceDirectory + '/y.txt". Please check source directory.')
    else:
        pass
    xMixturesTest = os.path.isfile(sourceDirectory + '/X_mixtures.txt')
    if(xMixturesTest == False):
        raise ValueError('Error: X_Mixtures dataset does not exist. Expected: "' + sourceDirectory + '/X_mixtures.txt". Please check source directory.')
    else:
        pass

def generateTestingParameters(parameters = None, verbose = True):
    #Sets Default testing parameters for the fingerprint function.
    if(type(parameters) == str):
        parameters = parameters.lower()
    else:
        pass
    if(parameters == "default"):
        parameters = fingerprintParametersClass()
        # Default number of importance iterations to run.
        parameters.importance_iterations = 100
        #Default number of final iterations to run.
        parameters.final_iterations = 100
        #Default number of features to retain in diagnostic fingerprints.
        parameters.num_features = 10
        #Default fraction (percent) of samples to be left out of model development for testing.
        parameters.test_size = 0.5
        #Default number of combinations of tuning parameters:
        parameters.n_combos = 100
        #Default number of random state seeds to be used.
        #For each randomly chosen C value, the model performance is evaulated for "n_rs" different seeds.
        parameters.n_rs = 5

    elif(parameters == None):
        parameters = fingerprintParametersClass()
        while(parameters.importance_iterations == None):
            #Define number of importance iterations to run (Reccomend 100-1000).
            try:
                parameters.importance_iterations = int(input("Define number of importance iterations to run (Reccomend 100-1000): "))
            except:
                print("\nError: Invalid entry. Please enter an integer value.")
                parameters.importance_iterations = None
        #Define number of final iterations to run (Reccomend 100-1000).
        while(parameters.final_iterations == None):
            try:
                parameters.final_iterations = int(input("Define number of final iterations to run (Reccomend 100-1000): "))
            except:
                print("\nError: invalid entry. Please enter an interger value.")
                parameters.final_iterations = None
        #Define number of features to retain in diagnostic fingerprints.
        while(parameters.num_features == None):
            try:
                parameters.num_features = int(input("Define number of features to retain in diagnostic fingerprints: "))
            except:
                print("\nError: invalid entry. Please enter an integer value.")
                parameters.num_features = None
        #What percentage of the samples do you want to use for testing (i.e., holdout) dataset?
        #the fraction of samples to be left out of model development for testing, needed for "train-test split"
        while(parameters.test_size == None):
            try:
                parameters.test_size = float(input("Define percentage of the samples to use for testing, i.e. holdout (Between 0 and 1): "))
                if(parameters.test_size >= 1):
                    print("\nError: invalid entry. Please enter a decimal value less than 1.")
                    parameters.test_size = None
            except:
                print("\nError: invalid entry. Please enter a decimal value less than 1.")
                parameters.test_size = None
        #define ranges for hyperparameters to be subsampled from
        #reccomend 100-1000. Number of combinations of tuning parameters:
        while(parameters.n_combos == None):
            try:
                parameters.n_combos = int(input("Define number of combinations of tuning parameters (recommended 100-1000): "))
            except:
                print("\nError: invalid entry. Please enter an integer value.")
                parameters.n_combos = None
        while(parameters.n_rs == None):
            try:
                parameters.n_rs = int(input("Define number of random state seeds to be used.: "))
            except:
                print("\nError: invalid entry. Please enter an integer value.")
                parameters.n_rs = None
                
    elif(parameters == fingerprintParametersClass):
        #Test if provided class object has valid parameters.
        #Define number of importance iterations to run (Reccomend 100-1000).
        try:
            parameters.importance_iterations = int(parameters.importance_iterations)
        except:
            raise ValueError('Error: invalid entry for parameter importance_iterations: "' + str(parameters.importance_iterations) +'". Please use an integer value and rerun script.')
        #Define number of final iterations to run (Reccomend 100-1000).
        try:
            parameters.final_iterations = int(parameters.final_iterations)
        except:
            raise ValueError('Error: invalid entry for parameter final_iterations: "' + str(parameters.final_iterations) +'". Please enter an interger value and rerun script.')
        try:
            parameters.num_features = int(parameters.num_features)
        except:
            raise ValueError('Error: invalid entry for parameter num_features: "' + str(parameters.num_features) + '". Please enter an integer value and rerun script.')
        #What percentage of the samples do you want to use for testing (i.e., holdout) dataset?
        #the fraction of samples to be left out of model development for testing, needed for "train-test split"
        try:
            parameters.test_size = float(parameters.test_size)
            if(parameters.test_size >= 1):
                raise ValueError('Error: invalid entry for parameter test_size: "' + str(parameters.test_size) +'". Please enter a decimal value less than 1 and rerun script.')
            else:
                pass
        except:
            raise ValueError('Error: invalid entry for parameter test_size: "' + str(parameters.test_size) +'". Please enter a decimal value less than 1 and rerun script.')
        #define ranges for hyperparameters to be subsampled from
        try:
            parameters.n_combos = int(parameters.n_combos)
        except:
            raise ValueError('Error: invalid entry for parameter n_combos: "' + str(parameters.n_combos) + '". Please enter an integer value and rerun script.')
        try:
            parameters.n_rs = int(parameters.n_rs)
        except:
            raise ValueError('Error: invalid entry for parameter n_rs: "' + str(parameters.n_rs) + '". Please enter an integer value and rerun script.')
    else:
        raise ValueError('"Error: parameter input: "' + parameters + '" is not valid. Please use a fingerprintingParametersClass object, or "default".')        
    if(verbose == True):
        print("Testing parameters:")
        print("II " + str(parameters.importance_iterations))
        print("FI " + str(parameters.final_iterations))
        print("NF " + str(parameters.num_features))
        print("TS " + str(parameters.test_size))
        print("NC " + str(parameters.n_combos))
        print("NRS " + str(parameters.n_rs))
    else:
        pass
    return parameters

def generateSVCParameters(parametersSVC = None, verbose = True):
    if(type(parametersSVC) == str):
        parametersSVC = parametersSVC.lower()
    else:
        pass
    if(parametersSVC == "default"):
        parametersSVC = fingerprintSVCParametersClass()
        #Classifier tuning parameter lower limit.
        parametersSVC.c_lower = 0.00001
        #Classifier tuning parameter upper limit.
        parametersSVC.c_upper = 10
        #Classifier tuning parameter number of divisions within range.
        parametersSVC.c_divisions = 100000000
    elif(parametersSVC == None):
        parametersSVC = fingerprintSVCParametersClass()
        while(parametersSVC.c_lower == None):
            try:
                parametersSVC.c_lower = float(input("Define lower limit of C (SVC classifier tuning parameter) "))
            except:
                print("\nError: invalid entry. Please enter a float value.")
                parametersSVC.c_lower = None
        while(parametersSVC.c_upper == None):
            try:
                parametersSVC.c_upper = float(input("Define upper limit of C (SVC classifier tuning parameter) "))
            except:
                print("\nError: invalid entry. Please enter a float value.")
                parametersSVC.c_upper = None
        #Not required?
        #while(parametersSVC.c_divisions == None):
            #try:
                #parametersSVC.c_divisions = int(input("Define number of divisions in range for SVC classifier tuning parameter: "))
            #except:
                #print("\nError: invalid entry. Please enter an integer value.")
                #parametersSVC.c_divisions = None
    elif(parametersSVC == fingerprintSVCParametersClass):
        try:
            parametersSVC.c_lower = float(parametersSVC.c_lower)
        except:
            raise ValueError("\nError: invalid entry. Please enter a float value.")
        try:
            parametersSVC.c_upper = float(parametersSVC.c_upper)
        except:
            raise ValueError("\nError: invalid entry. Please enter a float value.")
    else:
        raise ValueError('Error! Value "'+ parametersSVC + '" is not valid. Please create a fingerprintSVCParametersClass to pass into the function.')
        
    return parametersSVC

def importFingerprintData():
    import pandas as pd
    import numpy as np
    ### Import data
    #Import the X (i.e., predictor) dataset
    X = pd.read_csv('X.txt', header = None, delimiter = '\t')
    #Data is log transformed to increase normality of the data.
    X = np.log10(X+1)
    #use scikit learn's algorithim for normalizing the data from 0-1 (not currently in use)                
    #X = preprocessing.normalize(X, axis = 0)
    
    #Transpose samples from columns to rows                    
    X = X.T
    
    #Import the y (i.e., categorical) dataset. 
    #These data should "pure" sources without mixtures.
    #Only a single predictor varaible can be used at once
    #These data need to have the same number of samples as the X dataset
    y = pd.read_csv('y.txt', header = None, delimiter = '\t')
    #Transpose samples from columns to rows   
    y = y.T
    #Returns a contiguous flattened array                                                   
    y = np.ravel(y)        
    
    #Import mixture dataset. This data is expected to contain some mixture of multiple sources from the y dataset
    #This dataset need to have the same number of features as the X dataset
    #The features need to be in the same order as the X dataset
    #The mixture dataset can have different numbers of samples
    X_mixtures = pd.read_csv('X_mixtures.txt', header = None, delimiter = '\t')
    X_mixtures = np.log10(X_mixtures+1)
    #Transpose samples from columns to rows   
    X_mixtures = X_mixtures.T
    
    #create a unique index for each feature
    FeatureIndex = np.arange(1,len(X.T)+1,1)
    #FeatureIndex = pd.read_csv('feature_names.txt', header = None, delimiter = '\t')
    #FeatureIndex = np.ravel(FeatureIndex)
    
    ########################################################
    ### Check to see if X and y are the same length
    ### Check to see if the X and X_mixture datasets have the same number of features
    ########################################################
    
    
    if (X.shape[0] != y.shape[0]):
        print('\nERROR: x and y do not have the same number of samples.\n')
        print('X=', X.shape[0], 'samples')
        print('y=', y.shape[0], 'samples')
        raise ValueError('X and y do not have the same number of samples')
    else:
        pass
    
    if X.shape[1] != X_mixtures.shape[1]:
        print("\nERROR: X and X_mixtures do not have the same number of chemical features\
                 \n         X = " + str(X.shape[1]) + " chemical features\
                 \nX_mixtures = " + str(X_mixtures.shape[1]) + " chemical features")
        raise ValueError('X and X_mixtures must have the same number of chemical features.')
    else:
        pass
    return X, y, X_mixtures, FeatureIndex

#Create arrays needed to run and store information for each ML model
def misc_ml_storing_parameters(algorithm, n_combos):
    
    if algorithm == "SVC":
        sParameters = storingParametersClass()
        sParameters.classifier = 2
        sParameters.results = np.zeros((sParameters.classifier,n_combos))
        return (sParameters)

    else:
        print('whoops...undefined algorithm selected')

#set the domain of relevant tuning parameters which will be randomly drawn from
def set_hyperparameter_domain(algorithm, hyperParameterConstraints):

    if algorithm == "SVC":
        #c=tuning parameter in Classifier. The first number is the lower limit, the second number is the upper limit, and the third number is the number of divisions within this range.
        #C_ = np.geomspace(0.000001,1000,num=100000,endpoint=True)
        #C_ = np.geomspace(parameters.c_lower,parameters.c_upper,parameters.c_divisions,endpoint=True)
        C_ = np.geomspace(hyperParameterConstraints.c_lower, hyperParameterConstraints.c_upper, num=100000000, endpoint=True)
        
        return(C_)

    else:
        print('whoops...undefined algorithm selected')

#select a hyperparameter tuning value
def set_hyperparameter_value(algorithm, hyperparameter_domain):
    
    if algorithm == "SVC":
        C_ = hyperparameter_domain[:]
        C = random.choice(C_)  
        return(C)
    else:
        print('whoops...undefined algorithm selected')

#define the classifier to be used in the workflow 
def Classifier(algorithm, hyperparameter_value):
    
    if algorithm == "SVC":
        # assign hyperparameter values
        C = hyperparameter_value

        #import ML_algorithm from sklearn
        from sklearn.svm import SVC
        if algorithm == "SVC":
            estimator = SVC(kernel='linear',
                    C=C, 

                    tol=0.00001, 
                    shrinking=True, 
                    cache_size=200, 
                    verbose=False, 
                    max_iter=-1, 
                    probability=True)

        return (estimator)
    else:
        print('whoops...undefined algorithm selected')




#define the classifier to be used in the workflow 
def Classifier2(algorithm, hyperparameter_value, X_train, X_test, y_train, y_test, X, X_mixtures):
    
    if algorithm == "SVC":
        # assign hyperparameter values
        C = hyperparameter_value

        #import ML_algorithm from sklearn
        from sklearn.svm import SVC
        if algorithm == "SVC":
            estimator = SVC(kernel='linear',
                    C=C, 

                    tol=0.00001, 
                    shrinking=True, 
                    cache_size=200, 
                    verbose=False, 
                    max_iter=-1, 
                    probability=True)
            
        #The Classifier fits the training and testing datastudent be supervised by a 
        estimator.fit(X_train,y_train)
        #predicted y data based on the modeled X_train data
        y_train_pred = estimator.predict(X_train)
        #predicted y data based on the modeled X_test data
        y_test_pred = estimator.predict(X_test)                                   
        ########################################################################
        #Evaluate the performance of the model for the training data
        ########################################################################
        #the confusion matrix identifies the true positive ([0,0]), true negative ([1,1]), false positive ([0,1]), and false negative ([1,0])
        train_matrix = confusion_matrix(y_train, y_train_pred)
        #identifies the percentage of ture positives
        True_pos_train = train_matrix[1,1]
        #identifies the percentage of false negatives
        False_neg_train = train_matrix[1,0]
        #identifies the percentage of true negatives
        True_neg_train = train_matrix[0,0]
        #identifies the percentage of false positives
        False_pos_train = train_matrix[0,1]

        TrainAccuracy = 0.5*(True_pos_train/(True_pos_train+False_neg_train)+True_neg_train/(True_neg_train+False_pos_train))

        #######################################################################
        #Evaluate the performance of the model for the testing data
        ########################################################################
        #same as above, but for testing data
        test_matrix = confusion_matrix(y_test, y_test_pred)           
        #identifies the percentage of ture positives
        True_pos_test = test_matrix[1,1]
        #identifies the percentage of false negatives
        False_neg_test = test_matrix[1,0]
        #identifies the percentage of true negatives
        True_neg_test = test_matrix[0,0]
        #identifies the percentage of false positives
        False_pos_test = test_matrix[0,1]

        TestAccuracy = 0.5*(True_pos_test/(True_pos_test+False_neg_test)+True_neg_test/(True_neg_test+False_pos_test))

        ########################################################################
        #Classifier outputs
        ########################################################################
        #save probability of group membership
        proba_ = estimator.predict_proba(X)
        #save probabilities of group membership for mixture samples
        Mix_proba_ = estimator.predict_proba(X_mixtures)
        #predict group membership for all samples (not just test train splits)
        pred_all = estimator.predict(X)  
        
        return(y_test_pred, TrainAccuracy, TestAccuracy, estimator.coef_, proba_, Mix_proba_, pred_all)


        #return (estimator)
    else:
        print('whoops...undefined algorithm selected')

#Balanced Accuracy is used to assess the performance of each model in order to compare the cross validation performance following test-train split and tuning. This function identifies the "best" performing model based on the training and testing R2.
def test_metric(TrainAccuracy,TestAccuracy):
    #set performance threshold (the difference between training and testing R2...recommend <= 0.1). This is to help reduce overfitting
    threshold = 0.1
    #A: The higer the testing performance (i.e., the independent measure of model performance) the better the model. Thus, this is the baseline value of the metric score.
    A = TestAccuracy
    #B: Metric score penalty parameter- when the difference between training and testing performance is > threshold, the metric score is automatically 0. Overfitting is identified when the testing performance is much lower than the training performance.
    if (TrainAccuracy-TestAccuracy) > threshold:                 
        B = 0
    else:
        B = 1
    #C: Metric score penalty parameter- if the training or testing performance = 0, the test metric is automatically set to 0.
    if TrainAccuracy*TestAccuracy == 0:                         
        C = 0
    else:
        C = 1
    # actual metric that incorporates conditional statements
    metric = A*B*C    
    return(metric)
        
 #print the particular value of the randomly selected hyperparameter value and print the average balanced accuracy
def model_printout(algorithm, hyperparameter_value, avg_score):
    if algorithm == "SVC":
        print("     C = ", "{:.1e}".format(hyperparameter_value))
        print("     Averaged Balanced Accuracy of the Testing data = ", np.round(avg_score,2))       
        
        
#execute the model. This script selects hyper parameters, performs train test split, calculated balanced accuracy, and generates model outputs to be used later 
def model_execute(algorithm, X, y, X_mixtures, n_combos, n_rs, test_size_tuning, hyperParameterConstraints):
    i = 0 
    results = misc_ml_storing_parameters(algorithm, n_combos)
    
    #set domain of relevant tuning parameters
    hyperparameter_domain = set_hyperparameter_domain(algorithm, hyperParameterConstraints)

    #select tuning parameter value
    hyperparameter_value = set_hyperparameter_value(algorithm, hyperparameter_domain)
    
    #Iterate n times
    while i < n_combos:                                                                         
    #Classifier requires tuning of different parameters.
    #Pick a random hyper-parameter value within the defined range

        #Create empty arrays to store balanced accuracy
        metric_score = np.zeros(n_rs)
        
        print("Iteration:", i+1, "of", n_combos)
        start = timeit.default_timer()
        try:
            #Iterate through each random state value
            for u in range(0,n_rs):

                # apply train test split
                #Randomly split training and testing data. However, each combination will have the same train/test split based on random_state=rand_s[u]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_tuning,random_state=None,shuffle=True)
                #The Classifier fits the training and testing datastudent be supervised by a 
                estimator = Classifier(algorithm, hyperparameter_value)
                estimator.fit(X_train,y_train)
                #predicted y data based on the modeled X_train data
                y_train_pred = estimator.predict(X_train)
                #predicted y data based on the modeled X_test data
                y_test_pred = estimator.predict(X_test)                                   
                ########################################################################
                #Evaluate the performance of the model for the training data
                ########################################################################
                #the confusion matrix identifies the true positive ([0,0]), true negative ([1,1]), false positive ([0,1]), and false negative ([1,0])
                train_matrix = confusion_matrix(y_train, y_train_pred)
                #identifies the percentage of ture positives
                True_pos_train = train_matrix[1,1]
                #identifies the percentage of false negatives
                False_neg_train = train_matrix[1,0]
                #identifies the percentage of true negatives
                True_neg_train = train_matrix[0,0]
                #identifies the percentage of false positives
                False_pos_train = train_matrix[0,1]

                TrainAccuracy = 0.5*(True_pos_train/(True_pos_train+False_neg_train)+True_neg_train/(True_neg_train+False_pos_train))

                #######################################################################
                #Evaluate the performance of the model for the testing data
                ########################################################################
                #same as above, but for testing data
                test_matrix = confusion_matrix(y_test, y_test_pred)           
                #identifies the percentage of ture positives
                True_pos_test = test_matrix[1,1]
                #identifies the percentage of false negatives
                False_neg_test = test_matrix[1,0]
                #identifies the percentage of true negatives
                True_neg_test = test_matrix[0,0]
                #identifies the percentage of false positives
                False_pos_test = test_matrix[0,1]

                TestAccuracy = 0.5*(True_pos_test/(True_pos_test+False_neg_test)+True_neg_test/(True_neg_test+False_pos_test))

                ########################################################################
                #Classifier outputs
                ########################################################################
                #save probability of group membership
                proba_ = estimator.predict_proba(X)
                #save probabilities of group membership for mixture samples
                Mix_proba_ = estimator.predict_proba(X_mixtures)
                #predict group membership for all samples (not just test train splits)
                pred_all = estimator.predict(X)
                #Apply metric score to evaluate overfitting
                metric_score[u] = test_metric(TrainAccuracy,TestAccuracy)
    
        except:
            #Especially for sources with few samples, there is a chance that the training dataset has only one group (only absences are present). This will throw an error. THe try/except statment was included to prevent the script from stopping.
            print('     Error: Rerunning...')
            
        #Calculate the average metric score...sometimes, the metric score is nan (unsure why), so nanmean is used to ignore these "values" 
        avg_score = np.nanmean(metric_score)
        #Store average metric score and corresponding params  
        results.results[:,i] = np.array([avg_score,hyperparameter_value])
        model_printout(algorithm, hyperparameter_value, avg_score)
        stop = timeit.default_timer() 
        print("     Time = ", np.round(stop - start, decimals = 0), "seconds")
        i = i+1
           
    return (results)   

#TESTING

#execute the model. This script selects hyper parameters, performs train test split, calculated balanced accuracy, and generates model outputs to be used later 
def model_execute2(algorithm, X, y, X_mixtures, n_combos, n_rs, test_size_tuning, hyperParameterConstraints, hyperparameter_domain, verbose):
    i = 0 
    results = misc_ml_storing_parameters(algorithm, n_combos)
    
    #Iterate n times
    while i < n_combos:                                                                         
    #Classifier requires tuning of different parameters.
  
        #select tuning parameter value
        hyperparameter_value = set_hyperparameter_value(algorithm, hyperparameter_domain)
    
        #Create empty arrays to store balanced accuracy
        metric_score = np.zeros(n_rs)
        
        if(verbose == True):
            print("Iteration:", i+1, "of", n_combos)
        else:
            pass
        start = timeit.default_timer()
        #try:
        #Iterate through each random state value
        for u in range(0,n_rs):

            # apply train test split
            #Randomly split training and testing data. However, each combination will have the same train/test split based on random_state=rand_s[u]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_tuning,random_state=None,shuffle=True)
            #print(X_train)
            #The Classifier fits the training and testing datastudent be supervised by a 
            y_test_pred, TrainAccuracy, TestAccuracy, coef_, proba_, Mix_proba_, pred_all = Classifier2(algorithm, hyperparameter_value, X_train, X_test, y_train, y_test, X, X_mixtures)
            #Apply metric score to evaluate overfitting
            metric_score[u] = test_metric(TrainAccuracy,TestAccuracy)
    
        #except:
            #raise
            #Especially for sources with few samples, there is a chance that the training dataset has only one group (only absences are present). This will throw an error. THe try/except statment was included to prevent the script from stopping.
            #print('     Error: Rerunning...')
            
        #Calculate the average metric score...sometimes, the metric score is nan (unsure why), so nanmean is used to ignore these "values" 
        avg_score = np.nanmean(metric_score)
        #Store average metric score and corresponding params
        results.results[:,i] = np.array([avg_score,hyperparameter_value])
        if(verbose == True):
            model_printout(algorithm, hyperparameter_value, avg_score)
        else:
            pass
        stop = timeit.default_timer() 
        if(verbose == True):
            print("     Time = ", np.round(stop - start, decimals = 0), "seconds")
        else:
            pass
        i = i+1
           
    return (results)

#execute the model. This script selects hyper parameters, performs train test split, calculated balanced accuracy, and generates model outputs to be used later 
def model_execute22(algorithm, X, y, X_mixtures, n_combos, n_rs, test_size_tuning, hyperParameterConstraints):
    i = 0 
    results = misc_ml_storing_parameters(algorithm, n_combos)
    
    #set domain of relevant tuning parameters
    hyperparameter_domain = set_hyperparameter_domain(algorithm, hyperParameterConstraints)

    #select tuning parameter value
    hyperparameter_value = set_hyperparameter_value(algorithm, hyperparameter_domain)
    
    #Iterate n times
    while i < n_combos:                                                                         
    #Classifier requires tuning of different parameters.
    #Pick a random hyper-parameter value within the defined range

        #Create empty arrays to store balanced accuracy
        metric_score = np.zeros(n_rs)
        
        print("Iteration:", i+1, "of", n_combos)
        start = timeit.default_timer()
        try:
            #Iterate through each random state value
            for u in range(0,n_rs):

                # apply train test split
                #Randomly split training and testing data. However, each combination will have the same train/test split based on random_state=rand_s[u]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_tuning,random_state=None,shuffle=True)
                #The Classifier fits the training and testing datastudent be supervised by a 
                y_test_pred, TrainAccuracy, TestAccuracy, coef_, proba_, Mix_proba_, pred_all = Classifier2(algorithm, hyperparameter_value, x_train, x_test, y_train, y_test, X, X_mixtures)
                #Apply metric score to evaluate overfitting
                metric_score[u] = test_metric(TrainAccuracy,TestAccuracy)
    
        except:
            raise
            #Especially for sources with few samples, there is a chance that the training dataset has only one group (only absences are present). This will throw an error. THe try/except statment was included to prevent the script from stopping.
            print('     Error: Rerunning...')
            
        #Calculate the average metric score...sometimes, the metric score is nan (unsure why), so nanmean is used to ignore these "values" 
        avg_score = np.nanmean(metric_score)
        #Store average metric score and corresponding params  
        results.results[:,i] = np.array([avg_score,hyperparameter_value])
        model_printout(algorithm, hyperparameter_value, avg_score)
        stop = timeit.default_timer() 
        print("     Time = ", np.round(stop - start, decimals = 0), "seconds")
        i = i+1
           
    return (results)


#this script rearranges the scores, removes nan values, sorts the top scores, and saves the output.
def model_results(algorithm, results, verbose):
    resultsOut = modeledResultsClass()
    #Transpose results array to store by column.
    results_T = results.T
    
    #Sort the scores and keep corresponding params.
    resultsOut.resultsSorted = results_T[results_T[:,0].argsort()[::-1]]
    #Sometimes, a nan is present in the results data. This removes nan values from consideration.
    
    #Testing
    #resultsOut.resultsSorted = resultsOut.resultsSorted[~np.isnan(resultsOut.resultsSorted).any(axis=1)]
    #Retrieve the top performing parameters.
    score_top = resultsOut.resultsSorted[0,0]
    #Determine path based on chosen algorithm.
    if algorithm == "SVC":
        
        resultsOut.label = (['Metric Score', 'C'])
        C_top = resultsOut.resultsSorted[0,1]
        resultsOut.bestParamsTop = [round(score_top,5), round(C_top,5)]
        if(verbose == True):
            print()
            print('Best score = ',resultsOut.bestParamsTop[0])
            print('     C_top = ', resultsOut.bestParamsTop[1])
            print()
        else:
            pass
        
        resultsOut.hyperParameterTop = C_top
        return resultsOut

#This script
def plot_results(tuning, algorithm, modeledResults, IDName, n_params_Classifier):
    if algorithm == "SVC":
        #Print the parameter distributions againts the metric score
        for i in range(0,n_params_Classifier):
            #Itreatively plot parameters against score
            params_to_plot = modeledResults.resultsSorted[:,i]
            #Extract the scores
            scores_to_plot = modeledResults.resultsSorted[:,0]
            plt.figure(figsize=(8,5))
            plt.plot(params_to_plot[scores_to_plot>=0],scores_to_plot[scores_to_plot>=0],'.b')     
            plt.rc('axes', labelsize=20)
            plt.rc('xtick', labelsize=20)
            plt.rc('ytick', labelsize=20)
            plt.plot(modeledResults.resultsSorted[0:10,i],modeledResults.resultsSorted[0:10,0],'.r')
            plt.plot(modeledResults.resultsSorted[0,i],modeledResults.resultsSorted[0,0],'.c')  
            plt.xlabel(modeledResults.label[i])
            plt.ylabel('Balanced Accuracy')
            plt.xscale('log')
            filename = 'Final Results SVC\C_'+tuning+'_'+IDName+'_'+modeledResults.label[i]+'.png'
            plt.savefig(filename, dpi=1000, bbox_inches="tight")
            plt.show()


#Normalized Importance Function. This function takes the output of Classifier_CV_Imp and normalizes it for later use
def Norm_Importance(imp):
    normImpMetrics = importanceMetricsClass()
    # average ranking [after i iterations] of model j and store it in an array
    mean_imp = np.nanmean (imp,axis = 0)
    # stdev ranking [after i iterations] of model j and store it in an array
    std_imp = np.nanstd (imp, axis = 0)
    abs_mean_imp = abs(mean_imp)
    normImpMetrics.impMean = mean_imp/(abs_mean_imp.max())
    normImpMetrics.impStd = std_imp/(abs_mean_imp.max())
    normImpMetrics.absMean = abs_mean_imp/(abs_mean_imp.max())

    return (normImpMetrics)

#importance
def importance(algorithm, X, y, X_mixtures, test_size, importance_iterations, hyperparameter_top):
    #creates empty arrays for storing the mean and stdev varaible importance
    mean_imp = np.zeros([len(X.T)])
    std_imp = np.zeros([len(X.T)])
    #Creats empty array for storeing the importance of the ith model, jth run (rows = # of iterations, columns =# of vars.) .
    imp = np.zeros([importance_iterations, len(X.T)])

    #The script below calculates the Classifier coefficient/importance for each feature.
    #Start a timer to keep track of how long the entire process takes
    all_start = timeit.default_timer()                                                   

    i = 0
    j = 0
    k = 0
    #A while loop is used. Anytime a presence/absence value is not within the training dataset, an error occurs. The while loop allows us to move past the error and continue.
    while i < importance_iterations:
        #Start a timer to keep track of how long each iteration takes
        start = timeit.default_timer()
        try:
            if algorithm == "SVC":
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=None)

                #execute the Classifier_CV_Imp function                              
                #estimator = Classifier2
                estimator = Classifier (algorithm, hyperparameter_top)
                estimator.fit(X_train,y_train)
                y_train_pred = estimator.predict(X_train) 
                y_test_pred = estimator.predict(X_test)

                ########################################################################
                #Evaluate the performance of the model for the training data
                ########################################################################
                #the confusion matrix identifies the true positive ([0,0]), true negative ([1,1]), false positive ([0,1]), and false negative ([1,0])
                train_matrix = confusion_matrix(y_train, y_train_pred)
                #identifies the percentage of ture positives
                True_pos_train = train_matrix[1,1]
                #identifies the percentage of false negatives
                False_neg_train = train_matrix[1,0]

                #identifies the percentage of ture negatives
                True_neg_train = train_matrix[0,0]
                #identifies the percentage of false positives
                False_pos_train = train_matrix[0,1]                           

                TrainAccuracy = 0.5*(True_pos_train/(True_pos_train+False_neg_train)+True_neg_train/(True_neg_train+False_pos_train))

                #######################################################################
                #Evaluate the performance of the model for the testing data
                ########################################################################

                test_matrix = confusion_matrix(y_test, y_test_pred)

                True_pos_test = test_matrix[1,1]
                False_neg_test = test_matrix[1,0]

                True_neg_test = test_matrix[0,0]
                False_pos_test = test_matrix[0,1]

                TestAccuracy = 0.5*(True_pos_test/(True_pos_test+False_neg_test)+True_neg_test/(True_neg_test+False_pos_test))

                ########################################################################
                #Classifier outputs
                ########################################################################
                proba_ = estimator.predict_proba(X)
                Mix_proba_ = estimator.predict_proba(X_mixtures)
                pred_all = estimator.predict(X)
                coef_ = estimator.coef_

                if TrainAccuracy <= 0.5 or TestAccuracy <= 0.5:
                    stop = timeit.default_timer() 
                    print('Accuracy threshold failure')
                    print("     Balanced Training Accuracy: ", np.round(TrainAccuracy,2))
                    print("     Balanced Testing Accuracy:  ", np.round(TestAccuracy,2))
                    print('     n = ', importance_iterations - j, ' failures remaining')
                    print("     Time = ", np.round(stop - start, decimals = 0), "seconds")
                    print()
                    j = j+1
                    if j == importance_iterations:
                        print('Exceeded accuracy threshold failure limit')
                        break
                    else:
                        continue
                else:
                    #Stop iteration timer
                    stop = timeit.default_timer()                                                 
                    print("Iteration:", i+1, "of", importance_iterations)
                    print("     Balanced Training Accuracy: ", np.round(TrainAccuracy,2))
                    print("     Balanced Testing Accuracy:  ", np.round(TestAccuracy,2))
                    print("     Time = ", np.round(stop - start, decimals = 0), "seconds")
                    print()
                    #Store the ranking in the jth row of importance array
                    imp[i,:] = coef_                                                              
                i = i+1
        except:
            print('     Error: Rerunning...')
            k = k+1
            if k == importance_iterations:
                print('Presence/Absence failure limit exceeded') 
                break
            else:
                continue
    #execute Importance function
    normImpMetrics = Norm_Importance(imp)
    normImpMetrics.importance = imp
    all_stop = timeit.default_timer()
    print('Total Time: ', np.round((all_stop - all_start)/60, decimals = 0), "Minutes")
    return (normImpMetrics)

#saving and plotting
def saving_plotting_imp(algorithm, IDName, X, X_mixtures, importanceMetrics,FeatureIndex):
    #Saves the scaled mean importance of each chemical feature
    np.savetxt('Final Results SVC\\'+algorithm+'_scaled_mean_coef_'+IDName+'_.txt', importanceMetrics.impMean, delimiter = '\t')
    #Saves the scaled standard devaition of the importance of each chemical feature
    np.savetxt('Final Results SVC\\'+algorithm+'_scaled_std_coef_'+IDName+'_.txt', importanceMetrics.impStd, delimiter = '\t')
    #Combines the chemical feature ordering index, the mean importance, and the chemical feature data 
    ForSorting = np.ma.row_stack((FeatureIndex, importanceMetrics.absMean, X))
    #Combines the chemical feature ordering index, the mean importance, and the chemical feature data
    ForSorting_pos_and_neg = np.ma.row_stack((FeatureIndex, importanceMetrics.impMean,importanceMetrics.absMean, X))
    #Saves the chemical feature ordering index, the mean importance, and the chemical feature data
    np.savetxt('Final Results SVC\\'+algorithm+'_ForSorting_'+IDName+'.txt', ForSorting_pos_and_neg, delimiter = '\t')
    #Combines the chemical feature ordering index, the mean importance, and the chemical feature data
    ForSortingMix = np.ma.row_stack((FeatureIndex, importanceMetrics.absMean, X_mixtures))                           
    #Saves scaled importance of each predictor variables for use with the unknown mixture samples
    np.savetxt('Final Results SVC\\'+algorithm+'ForSorting_mix_'+IDName+'.txt', ForSortingMix, delimiter = '\t')
    
    #print('Top feature names/IDs', ForSorting[0,0:num_features])
    
    #plot importance (by Cheng @02/01/2021)
    mean = importanceMetrics.impMean 
    std = importanceMetrics.impStd
    #confid_int_95 = 1.96*std/np.sqrt(100)
    confid_int_95 = std
    fig = plt.figure(figsize=(8,5))

    linewidth = 50000 
    plt.errorbar(range(len(mean)), mean[mean.argsort()], yerr=confid_int_95[mean.argsort()], fmt='-', c='r', ecolor='k', elinewidth=0.5,capsize=3, barsabove=False,capthick=.5, errorevery=1, linewidth = 3)
    plt.xlabel('Rank-Sorted Features by Importance', size=20)
    plt.ylabel('Feature Importance', size=20)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)

    #ax = plt.axes()
    #ax.set_yticks([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])

    plt.show()
    fig.savefig('Final Results SVC/'+algorithm+'_Importances_'+IDName+'_.png', dpi=300, bbox_inches='tight')
    return (ForSorting, ForSortingMix)  

#retuning with n featurs
def retuning(algorithm, y, test_size, n_combos, n_rs, num_features, ForSorting, ForSortingMix, hyperparameter_top, hyperParameterConstraints):
    
    ForSorting = ForSorting
    ForSortingMix = ForSortingMix
    
    #Rank sort the X (i.e., chemical data) based on the ranked importance
    X = ForSorting[ :, ForSorting[1].argsort()[::-1]]
    #Rank sort the X_mixtures (i.e., chemical data) based on the ranked importance              
    X_mixtures = ForSortingMix[ :, ForSortingMix[1].argsort()[::-1]]   

    #Retain only the n most important features from the rank sorted X data
    X = X[2:,0:num_features]
    #Retain only the n most important features from the rank sorted X_mixtures data                        
    X_mixtures = X_mixtures[2:,0:num_features]                          

    #transforming the test size into an integer for train-test split
    test_size_retuning = int(round(len(y)*test_size,0))
     
    results = misc_ml_storing_parameters(algorithm, n_combos)
   
    i = 0
    #Iterate n times
    while i < n_combos:
        #Classifier requires tuning of different parameters.
        #Pick a random hyper-parameter value within the defined range- defined in block 2
        #set range of relevant tuning parameters
        hyperparameter_domain = set_hyperparameter_domain(algorithm, hyperParameterConstraints)         

        #select tuning parameter value
        hyperparameter_value = set_hyperparameter_value(algorithm, hyperparameter_domain)
        #Create empty arrays to store data
        metric_score = np.zeros(n_rs)                                                           

        print("Iteration:", i+1, "of", n_combos)
        start = timeit.default_timer()
        try:
            #Iterate through each random state value
            for u in range(0,n_rs):

                # apply Classifier and metric score functions and store values
                #Randomly split training and testing data. However, each combination will have the same train/test split based on random_state=rand_s[u]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size_retuning,random_state=None,shuffle=True)
                #The Classifier fits the training and testing datastudent be supervised by a 
                #estimator = Classifier2 
                estimator = Classifier (algorithm, hyperparameter_value)
                estimator.fit(X_train,y_train)
                y_train_pred = estimator.predict(X_train) 
                y_test_pred = estimator.predict(X_test)

                ########################################################################
                #Evaluate the performance of the model for the training data
                ########################################################################
                #the confusion matrix identifies the true positive ([0,0]), true negative ([1,1]), false positive ([0,1]), and false negative ([1,0])
                train_matrix = confusion_matrix(y_train, y_train_pred)        
                #identifies the percentage of ture positives
                True_pos_train = train_matrix[1,1]
                #identifies the percentage of false negatives
                False_neg_train = train_matrix[1,0]                           
                #identifies the percentage of ture negatives
                True_neg_train = train_matrix[0,0]
                #identifies the percentage of false positives
                False_pos_train = train_matrix[0,1]                           

                TrainAccuracy = 0.5*(True_pos_train/(True_pos_train+False_neg_train)+True_neg_train/(True_neg_train+False_pos_train))

                #######################################################################
                #Evaluate the performance of the model for the testing data
                ########################################################################

                test_matrix = confusion_matrix(y_test, y_test_pred)

                True_pos_test = test_matrix[1,1]
                False_neg_test = test_matrix[1,0]

                True_neg_test = test_matrix[0,0]
                False_pos_test = test_matrix[0,1]

                TestAccuracy = 0.5*(True_pos_test/(True_pos_test+False_neg_test)+True_neg_test/(True_neg_test+False_pos_test))

                ########################################################################
                #Classifier outputs
                ########################################################################
                proba_ = estimator.predict_proba(X)
                Mix_proba_ = estimator.predict_proba(X_mixtures)
                pred_all = estimator.predict(X)
                #Apply metric score to evaluate overfitting
                metric_score[u] = test_metric(TrainAccuracy,TestAccuracy)

        except:
            print('     Error: Rerunning...')
        #Calculate the average metric score...sometimes, the metric score is nan (unsure why), so nanmean is used to ignore these "values" 
        avg_score = np.nanmean(metric_score)                                                    
        
        #Store score and corresponding params  
        results.results[:,i] = np.array([avg_score,hyperparameter_value])                                                  
        model_printout (algorithm, hyperparameter_value, avg_score)
        stop = timeit.default_timer() 
        print("     Time = ", np.round(stop - start, decimals = 0), "seconds")
        i = i+1

    return (results)

#retuning with n featurs
def retuning2(algorithm, y, test_size, n_combos, n_rs, num_features, ForSorting, ForSortingMix, hyperparameter_top, hyperParameterConstraints, hyperparameter_domain):
    
    ForSorting = ForSorting
    ForSortingMix = ForSortingMix
    
    #Rank sort the X (i.e., chemical data) based on the ranked importance
    X = ForSorting[ :, ForSorting[1].argsort()[::-1]]
    #Rank sort the X_mixtures (i.e., chemical data) based on the ranked importance              
    X_mixtures = ForSortingMix[ :, ForSortingMix[1].argsort()[::-1]]   
    #Retain only the n most important features from the rank sorted X data
    X = X[2:,0:num_features]
    #Retain only the n most important features from the rank sorted X_mixtures data                        
    X_mixtures = X_mixtures[2:,0:num_features]                          

    #transforming the test size into an integer for train-test split
    test_size_retuning = int(round(len(y)*test_size,0))
     
    results = misc_ml_storing_parameters(algorithm, n_combos)
   
    i = 0
    #Iterate n times
    while i < n_combos:
        #Classifier requires tuning of different parameters.
        #Pick a random hyper-parameter value within the defined range- defined in block 2

        #select tuning parameter value
        hyperparameter_value = set_hyperparameter_value(algorithm, hyperparameter_domain)
        #Create empty arrays to store data
        metric_score = np.zeros(n_rs)                                                           

        print("Iteration:", i+1, "of", n_combos)
        start = timeit.default_timer()
        try:
            #Iterate through each random state value
            for u in range(0,n_rs):

                # apply Classifier and metric score functions and store values
                #Randomly split training and testing data. However, each combination will have the same train/test split based on random_state=rand_s[u]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size_retuning,random_state=None,shuffle=True)
                #The Classifier fits the training and testing datastudent be supervised by a 
                #estimator = Classifier2 
                estimator = Classifier(algorithm, hyperparameter_value)
                estimator.fit(X_train,y_train)
                y_train_pred = estimator.predict(X_train) 
                y_test_pred = estimator.predict(X_test)

                ########################################################################
                #Evaluate the performance of the model for the training data
                ########################################################################
                #the confusion matrix identifies the true positive ([0,0]), true negative ([1,1]), false positive ([0,1]), and false negative ([1,0])
                train_matrix = confusion_matrix(y_train, y_train_pred)        
                #identifies the percentage of ture positives
                True_pos_train = train_matrix[1,1]
                #identifies the percentage of false negatives
                False_neg_train = train_matrix[1,0]                           
                #identifies the percentage of ture negatives
                True_neg_train = train_matrix[0,0]
                #identifies the percentage of false positives
                False_pos_train = train_matrix[0,1]                           

                TrainAccuracy = 0.5*(True_pos_train/(True_pos_train+False_neg_train)+True_neg_train/(True_neg_train+False_pos_train))

                #######################################################################
                #Evaluate the performance of the model for the testing data
                ########################################################################

                test_matrix = confusion_matrix(y_test, y_test_pred)

                True_pos_test = test_matrix[1,1]
                False_neg_test = test_matrix[1,0]

                True_neg_test = test_matrix[0,0]
                False_pos_test = test_matrix[0,1]

                TestAccuracy = 0.5*(True_pos_test/(True_pos_test+False_neg_test)+True_neg_test/(True_neg_test+False_pos_test))

                ########################################################################
                #Classifier outputs
                ########################################################################
                proba_ = estimator.predict_proba(X)
                Mix_proba_ = estimator.predict_proba(X_mixtures)
                pred_all = estimator.predict(X)
                #Apply metric score to evaluate overfitting
                metric_score[u] = test_metric(TrainAccuracy,TestAccuracy)

        except:
            print('     Error: Rerunning...')
        #Calculate the average metric score...sometimes, the metric score is nan (unsure why), so nanmean is used to ignore these "values" 
        avg_score = np.nanmean(metric_score)                                                    
        
        #Store score and corresponding params  
        results.results[:,i] = np.array([avg_score,hyperparameter_value])                                                  
        model_printout (algorithm, hyperparameter_value, avg_score)
        stop = timeit.default_timer() 
        print("     Time = ", np.round(stop - start, decimals = 0), "seconds")
        i = i+1

    return (results)



#rerunning with n diagnostic features
def diagnostic_rerun(algorithm, IDName, X, y, X_mixtures, test_size, final_iterations, hyperparameter_top):
    
    test_size_retuning = int(math.ceil(len(y)*test_size)) 
    
    #creates empty array for storing actual group membership of the testing dataset for each iteration
    test_actual = np.zeros((test_size_retuning,final_iterations))
    #creates empty array for storing predicted group membership of the testing dataset for each iteration
    test_pred = np.zeros((test_size_retuning,final_iterations))

    #"predict_proba" generates a probability of presence and absence, so 2 columns are needed for each sample (row)
    proba_known = np.zeros((len(y),2,final_iterations))
    #"predict_proba" generates a probability of presence and absence, so 2 columns are needed for each sample (row)          
    proba_mix = np.zeros((len(X_mixtures),2,final_iterations))
    #"predict_proba" generates a probability of presence and absence, so 2 columns are needed for each sample (row)
    pred_known =np.zeros((len(y),final_iterations))
    #"predict_proba" generates a probability of presence and absence, so 2 columns are needed for each sample (row)
    mean_proba_known = np.zeros((len(y),2))
    #"predict_proba" generates a probability of presence and absence, so 2 columns are needed for each sample (row)
    mean_proba_mix = np.zeros((len(X_mixtures),2))
    mean_pred_all = np.zeros(len(y))

    std_proba_known = np.zeros((len(y),2))
    std_proba_mix = np.zeros((len(X_mixtures),2))

    #start a timer to keep track of how long the entire process takes
    all_start = timeit.default_timer()                                 
    i = 0
    
   #start a timer to keep track of how long the entire process takes
   #iterate the improtance for i iterations
    while i < final_iterations:
        #start a timer to keep track of how long each iteration takes
        start = timeit.default_timer()
        try:
            if algorithm == "SVC":
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=None)
                
                #execute the Classifier_CV_Imp function                              
                estimator = Classifier(algorithm, hyperparameter_top)
                #estimator = Classifier2
                estimator.fit(X_train,y_train)
                y_train_pred = estimator.predict(X_train) 
                y_test_pred = estimator.predict(X_test)

                ########################################################################
                #Evaluate the performance of the model for the training data
                ########################################################################
                #the confusion matrix identifies the true positive ([0,0]), true negative ([1,1]), false positive ([0,1]), and false negative ([1,0])
                train_matrix = confusion_matrix(y_train, y_train_pred)

                #identifies the percentage of ture positives
                True_pos_train = train_matrix[1,1]
                #identifies the percentage of false negatives
                False_neg_train = train_matrix[1,0]
                #identifies the percentage of ture negatives
                True_neg_train = train_matrix[0,0]
                #identifies the percentage of false positives
                False_pos_train = train_matrix[0,1]

                TrainAccuracy = 0.5*(True_pos_train/(True_pos_train+False_neg_train)+True_neg_train/(True_neg_train+False_pos_train))

                #######################################################################
                #Evaluate the performance of the model for the testing data
                ########################################################################

                test_matrix = confusion_matrix(y_test, y_test_pred)

                True_pos_test = test_matrix[1,1]
                False_neg_test = test_matrix[1,0]

                True_neg_test = test_matrix[0,0]
                False_pos_test = test_matrix[0,1]

                TestAccuracy = 0.5*(True_pos_test/(True_pos_test+False_neg_test)+True_neg_test/(True_neg_test+False_pos_test))

                ########################################################################
                #Classifier outputs
                ########################################################################
                proba_ = estimator.predict_proba(X)
                Mix_proba_ = estimator.predict_proba(X_mixtures)
                pred_all = estimator.predict(X)
                coef_ = estimator.coef_

                proba_known [:,:,i] = proba_
                proba_mix [:,:,i] = Mix_proba_
                pred_known [:,i] = pred_all
               #stop iteration timer
                stop = timeit.default_timer()
                print("Iteration:", i+1, "of", final_iterations)
                print("     Balanced Training Accuracy: ", np.round(TrainAccuracy,2))
                print("     Balanced Testing Accuracy:  ", np.round(TestAccuracy,2))
                print("     Time = ", np.round(stop - start, decimals = 0), "seconds")
                print()
                test_pred[:,i] = y_test_pred
                test_actual[:,i] = y_test
                i = i+1
        except:
            print('Error: Rerunning...')
    
    for i in range(0,len(y),1):
        for j in range (0,2,1):
            mean_proba_known [i,j] = np.mean(proba_known [i,j,:])
            std_proba_known [i,j] = np.std(proba_known [i,j,:])
    CI_proba_known = 1.96*std_proba_known/(final_iterations)**0.5

    for i in range(0,len(X_mixtures),1):
        for j in range (0,2,1):
            mean_proba_mix [i,j] = np.mean(proba_mix [i,j,:])
            std_proba_mix [i,j] = np.std(proba_mix [i,j,:])

            CI_proba_mix = 1.96*std_proba_mix/(final_iterations)**0.5

    mean_pred_all = np.nanmean(pred_known, axis = 1)
    
    # The script below saves the relevant outputs as txt files
    np.savetxt('Final Results SVC\\Y_test_pred_CV_10_SVC_'+IDName+'_.txt', test_pred, delimiter = '\t')
    np.savetxt('Final Results SVC\\Y_test_actual_CV_10_SVC_'+IDName+'_.txt', test_actual, delimiter = '\t')
    np.savetxt('Final Results SVC\\mean_proba_known_10_SVC_'+IDName+'_.txt', mean_proba_known, delimiter = '\t')
    np.savetxt('Final Results SVC\\mean_proba_mix_10_SVC_'+IDName+'_.txt', mean_proba_mix, delimiter = '\t')
    np.savetxt('Final Results SVC\\std_proba_known_10_SVC_'+IDName+'_.txt', std_proba_known, delimiter = '\t')
    np.savetxt('Final Results SVC\\std_proba_mix_10_SVC_'+IDName+'_.txt', std_proba_mix, delimiter = '\t')
    np.savetxt('Final Results SVC\\CI_proba_known_10_SVC_'+IDName+'_.txt', CI_proba_known, delimiter = '\t')
    np.savetxt('Final Results SVC\\CI_proba_mix_10_SVC_'+IDName+'_.txt', CI_proba_mix, delimiter = '\t')
    
    all_stop = timeit.default_timer()
    print('Time: ', np.round((all_stop - all_start)/60, decimals = 0), "minutes")
    
    return(mean_pred_all, mean_proba_known, test_actual, test_pred)


#rerunning with n diagnostic features
def diagnostic_rerun2(algorithm, IDName, X, y, X_mixtures, test_size, final_iterations, hyperparameter_top):
    
    test_size_retuning = int(math.ceil(len(y)*test_size)) 
    
    #creates empty array for storing actual group membership of the testing dataset for each iteration
    test_actual = np.zeros((test_size_retuning,final_iterations))
    #creates empty array for storing predicted group membership of the testing dataset for each iteration
    test_pred = np.zeros((test_size_retuning,final_iterations))

    #"predict_proba" generates a probability of presence and absence, so 2 columns are needed for each sample (row)
    proba_known = np.zeros((len(y),2,final_iterations))
    #"predict_proba" generates a probability of presence and absence, so 2 columns are needed for each sample (row)          
    proba_mix = np.zeros((len(X_mixtures),2,final_iterations))
    #"predict_proba" generates a probability of presence and absence, so 2 columns are needed for each sample (row)
    pred_known =np.zeros((len(y),final_iterations))
    #"predict_proba" generates a probability of presence and absence, so 2 columns are needed for each sample (row)
    mean_proba_known = np.zeros((len(y),2))
    #"predict_proba" generates a probability of presence and absence, so 2 columns are needed for each sample (row)
    mean_proba_mix = np.zeros((len(X_mixtures),2))
    mean_pred_all = np.zeros(len(y))

    std_proba_known = np.zeros((len(y),2))
    std_proba_mix = np.zeros((len(X_mixtures),2))

    #start a timer to keep track of how long the entire process takes
    all_start = timeit.default_timer()                                 
    i = 0
    
   #start a timer to keep track of how long the entire process takes
   #iterate the improtance for i iterations
    while i < final_iterations:
        #start a timer to keep track of how long each iteration takes
        start = timeit.default_timer()
        try:
            if algorithm == "SVC":
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=None)
                
                #execute the Classifier_CV_Imp function                              
                estimator = Classifier(algorithm, hyperparameter_top)
                #estimator = Classifier2
                estimator.fit(X_train,y_train)
                y_train_pred = estimator.predict(X_train) 
                y_test_pred = estimator.predict(X_test)

                ########################################################################
                #Evaluate the performance of the model for the training data
                ########################################################################
                #the confusion matrix identifies the true positive ([0,0]), true negative ([1,1]), false positive ([0,1]), and false negative ([1,0])
                train_matrix = confusion_matrix(y_train, y_train_pred)

                #identifies the percentage of ture positives
                True_pos_train = train_matrix[1,1]
                #identifies the percentage of false negatives
                False_neg_train = train_matrix[1,0]
                #identifies the percentage of ture negatives
                True_neg_train = train_matrix[0,0]
                #identifies the percentage of false positives
                False_pos_train = train_matrix[0,1]

                TrainAccuracy = 0.5*(True_pos_train/(True_pos_train+False_neg_train)+True_neg_train/(True_neg_train+False_pos_train))

                #######################################################################
                #Evaluate the performance of the model for the testing data
                ########################################################################

                test_matrix = confusion_matrix(y_test, y_test_pred)

                True_pos_test = test_matrix[1,1]
                False_neg_test = test_matrix[1,0]

                True_neg_test = test_matrix[0,0]
                False_pos_test = test_matrix[0,1]

                TestAccuracy = 0.5*(True_pos_test/(True_pos_test+False_neg_test)+True_neg_test/(True_neg_test+False_pos_test))

                ########################################################################
                #Classifier outputs
                ########################################################################
                proba_ = estimator.predict_proba(X)
                Mix_proba_ = estimator.predict_proba(X_mixtures)
                pred_all = estimator.predict(X)
                coef_ = estimator.coef_

                proba_known [:,:,i] = proba_
                proba_mix [:,:,i] = Mix_proba_
                pred_known [:,i] = pred_all
               #stop iteration timer
                stop = timeit.default_timer()
                print("Iteration:", i+1, "of", final_iterations)
                print("     Balanced Training Accuracy: ", np.round(TrainAccuracy,2))
                print("     Balanced Testing Accuracy:  ", np.round(TestAccuracy,2))
                print("     Time = ", np.round(stop - start, decimals = 0), "seconds")
                print()
                test_pred[:,i] = y_test_pred
                test_actual[:,i] = y_test
                i = i+1
        except:
            print('Error: Rerunning...')
    
    for i in range(0,len(y),1):
        for j in range (0,2,1):
            mean_proba_known [i,j] = np.mean(proba_known [i,j,:])
            std_proba_known [i,j] = np.std(proba_known [i,j,:])
    CI_proba_known = 1.96*std_proba_known/(final_iterations)**0.5

    for i in range(0,len(X_mixtures),1):
        for j in range (0,2,1):
            mean_proba_mix [i,j] = np.mean(proba_mix [i,j,:])
            std_proba_mix [i,j] = np.std(proba_mix [i,j,:])

            CI_proba_mix = 1.96*std_proba_mix/(final_iterations)**0.5

    mean_pred_all = np.nanmean(pred_known, axis = 1)
    
    # The script below saves the relevant outputs as txt files
    np.savetxt('Final Results SVC\\Y_test_pred_CV_10_SVC_'+IDName+'_.txt', test_pred, delimiter = '\t')
    np.savetxt('Final Results SVC\\Y_test_actual_CV_10_SVC_'+IDName+'_.txt', test_actual, delimiter = '\t')
    np.savetxt('Final Results SVC\\mean_proba_known_10_SVC_'+IDName+'_.txt', mean_proba_known, delimiter = '\t')
    np.savetxt('Final Results SVC\\mean_proba_mix_10_SVC_'+IDName+'_.txt', mean_proba_mix, delimiter = '\t')
    np.savetxt('Final Results SVC\\std_proba_known_10_SVC_'+IDName+'_.txt', std_proba_known, delimiter = '\t')
    np.savetxt('Final Results SVC\\std_proba_mix_10_SVC_'+IDName+'_.txt', std_proba_mix, delimiter = '\t')
    np.savetxt('Final Results SVC\\CI_proba_known_10_SVC_'+IDName+'_.txt', CI_proba_known, delimiter = '\t')
    np.savetxt('Final Results SVC\\CI_proba_mix_10_SVC_'+IDName+'_.txt', CI_proba_mix, delimiter = '\t')
    
    all_stop = timeit.default_timer()
    print('Time: ', np.round((all_stop - all_start)/60, decimals = 0), "minutes")
    
    return(mean_pred_all, mean_proba_known, test_actual, test_pred)

#final plots...ROC, confusion matrix
def final_plots(y, IDName, mean_proba_known, test_actual, test_pred):
    #plot the ROC curve for the known data, averaged over n iterations
    print('Confusion Matrix for ALL observed data. This was generated by averaging the probability of group membership for all iterations')
    skplt.metrics.plot_roc(y, mean_proba_known, text_fontsize = 12)
    skplt.metrics.plot_confusion_matrix(y, np.argmax(mean_proba_known, axis=1), text_fontsize = 16, normalize=True)
    plt.title('Confusion matrix (all observed data)')
    plt.xlim(-0.5, len(np.unique(y))-0.5) 
    plt.ylim(len(np.unique(y))-0.5, -0.5) 
    plt.savefig('Final Results SVC\\SVC_Confusion_10_All_Obs_'+IDName+'_.png', dpi=300)
    print('Confusion Matrix for ALL cross validation data only. This was generated using the cross validation predictions for all iterations')
    skplt.metrics.plot_confusion_matrix(np.ravel(test_actual), np.ravel(test_pred), text_fontsize = 16, normalize=True)
    plt.title('Confusion matrix (CV data only)')
    plt.xlim(-0.5, len(np.unique(y))-0.5)
    plt.ylim(len(np.unique(y))-0.5, -0.5)
    plt.savefig('Final Results SVC\\SVC_Confusion_10_All_Cross_'+IDName+'_.png', dpi=300)