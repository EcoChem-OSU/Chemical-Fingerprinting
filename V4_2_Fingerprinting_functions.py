# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:37:34 2021

@author: Gerrad

"""

class fingerprintParametersClass():
    #Initial setup
    def __init__(self):
        self.importance_iterations = 100
        self.final_iterations = 100
        self.num_features = 10
        self.test_size = 0.5
        self.n_combos = 100
        self.n_rs = 2
        
    def generateParameters(self,parameters = None, verbose = True):
    #Sets Default testing parameters for the fingerprint function.
        
        if(type(parameters) == str):
            parameters = parameters.lower()
        else:
            pass
            
        if(parameters == "fast"):
            # Default number of importance iterations to run.
            self.importance_iterations = 10
            #Default number of final iterations to run.
            self.final_iterations = 10
            #Default number of features to retain in diagnostic fingerprints.
            self.num_features = 2
            #Default fraction (percent) of samples to be left out of model development for testing.
            self.test_size = 0.5
            #Default number of combinations of tuning parameters:
            self.n_combos = 10
            #Default number of random state seeds to be used.
            #For each randomly chosen C value, the model performance is evaulated for "n_rs" different seeds.
            self.n_rs = 2

        elif(parameters == None):
            self.importance_iterations = None
            self.final_iterations = None
            self.num_features = None
            self.test_size = None
            self.n_combos = None
            self.n_rs = None
            while(self.importance_iterations == None):
                #Define number of importance iterations to run (Reccomend 100-1000).
                try:
                    self.importance_iterations = int(input("Define number of importance iterations to run (Reccomend 100-1000): "))
                except:
                    print("\nError: Invalid entry. Please enter an integer value.")
                    self.importance_iterations = None
            #Define number of final iterations to run (Reccomend 100-1000).
            while(self.final_iterations == None):
                try:
                    self.final_iterations = int(input("Define number of final iterations to run (Reccomend 100-1000): "))
                except:
                    print("\nError: invalid entry. Please enter an interger value.")
                    self.final_iterations = None
            #Define number of features to retain in diagnostic fingerprints.
            while(self.num_features == None):
                try:
                    self.num_features = int(input("Define number of features to retain in diagnostic fingerprints: "))
                except:
                    print("\nError: invalid entry. Please enter an integer value.")
                    self.num_features = None
            #What percentage of the samples do you want to use for testing (i.e., holdout) dataset?
            #the fraction of samples to be left out of model development for testing, needed for "train-test split"
            while(self.test_size == None):
                try:
                    self.test_size = float(input("Define percentage of the samples to use for testing, i.e. holdout (Between 0 and 1): "))
                    if(self.test_size >= 1):
                        print("\nError: invalid entry. Please enter a decimal value less than 1.")
                        self.test_size = None
                except:
                    print("\nError: invalid entry. Please enter a decimal value less than 1.")
                    self.test_size = None
            #define ranges for hyperparameters to be subsampled from
            #reccomend 100-1000. Number of combinations of tuning parameters:
            while(self.n_combos == None):
                try:
                    self.n_combos = int(input("Define number of combinations of tuning parameters (recommended 100-1000): "))
                except:
                    print("\nError: invalid entry. Please enter an integer value.")
                    self.n_combos = None
            while(self.n_rs == None):
                try:
                    self.n_rs = int(input("Define number of random state seeds to be used.: "))
                except:
                    print("\nError: invalid entry. Please enter an integer value.")
                    self.n_rs = None
     
            if(verbose == True):
                print("\nTesting parameters:\n")
                print("Importance Iterations (importance_iterations) - " + str(self.importance_iterations))
                print("Final Iterations (final_iterations) - " + str(self.final_iterations))
                print("Number of Features (num_features) - " + str(self.num_features))
                print("Test Size (test_size) - " + str(self.test_size))
                print("Number of Combos (n_combos) - " + str(self.n_combos))
                print("Number of Random State Seeds (n_rs) - " + str(self.n_rs))
            else:
                pass
                    
    def validate(self):
            #Test if provided class object has valid self.
            #Define number of importance iterations to run (Reccomend 100-1000).
            try:
                self.importance_iterations = int(self.importance_iterations)
            except:
                raise ValueError('Error: invalid entry for parameter importance_iterations: "' + str(self.importance_iterations) +'". Please use an integer value and rerun script.')
            #Define number of final iterations to run (Reccomend 100-1000).
            try:
                self.final_iterations = int(self.final_iterations)
            except:
                raise ValueError('Error: invalid entry for parameter final_iterations: "' + str(self.final_iterations) +'". Please enter an interger value and rerun script.')
            try:
                self.num_features = int(self.num_features)
            except:
                raise ValueError('Error: invalid entry for parameter num_features: "' + str(self.num_features) + '". Please enter an integer value and rerun script.')
            #What percentage of the samples do you want to use for testing (i.e., holdout) dataset?
            #the fraction of samples to be left out of model development for testing, needed for "train-test split"
            try:
                self.test_size = float(self.test_size)
                if(self.test_size >= 1):
                    raise ValueError('Error: invalid entry for parameter test_size: "' + str(self.test_size) +'". Please enter a decimal value less than 1 and rerun script.')
                else:
                    pass
            except:
                raise ValueError('Error: invalid entry for parameter test_size: "' + str(self.test_size) +'". Please enter a decimal value less than 1 and rerun script.')
            #define ranges for hyperparameters to be subsampled from
            try:
                self.n_combos = int(self.n_combos)
            except:
                raise ValueError('Error: invalid entry for parameter n_combos: "' + str(self.n_combos) + '". Please enter an integer value and rerun script.')
            try:
                self.n_rs = int(self.n_rs)
            except:
                raise ValueError('Error: invalid entry for parameter n_rs: "' + str(self.n_rs) + '". Please enter an integer value and rerun script.')
    def show(self):
        print("\nTesting parameters:\n")
        print("Importance Iterations (importance_iterations) - " + str(self.importance_iterations))
        print("Final Iterations (final_iterations) - " + str(self.final_iterations))
        print("Number of Features (num_features) - " + str(self.num_features))
        print("Test Size (test_size) - " + str(self.test_size))
        print("Number of Combos (n_combos) - " + str(self.n_combos))
        print("Number of Random State Seeds (n_rs) - " + str(self.n_rs))

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

def train_test_split_grouped(X, y, test_size_tuning, size_restraint = False):
    #Performs SKLearns train_test_split using a percentage group of postive results and negative results, then combines those into single test and train arrays for x and y data.
    import numpy as np
    from sklearn.model_selection import train_test_split
    import random
    
    # Create a mask from the postive results from the y dummies
    mask = y>0
    X_trainpos, X_testpos, y_trainpos, y_testpos = train_test_split(X[mask], y[mask], test_size=test_size_tuning,random_state=None,shuffle=True)    
    X_trainneg, X_testneg, y_trainneg, y_testneg = train_test_split(X[np.invert(mask)], y[np.invert(mask)], test_size=test_size_tuning,random_state=None,shuffle=True)

    if((size_restraint != False) and (len(X_testpos) + len(X_testneg)) > size_restraint):
        n = random.randrange(0,(len(X_testneg)))
        X_trainneg = np.append(X_trainneg, X_testneg[n])
        X_testneg = np.delete(X_testneg,n)
    else:
        pass

    #Concatenate testing and training groups
    X_train = np.concatenate((X_trainneg, X_trainpos))
    X_test = np.concatenate((X_testneg, X_testpos))
    y_train = np.concatenate((y_trainneg, y_trainpos))
    y_test = np.concatenate((y_testneg, y_testpos))
    
    
    
    return X_train, X_test, y_train, y_test

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