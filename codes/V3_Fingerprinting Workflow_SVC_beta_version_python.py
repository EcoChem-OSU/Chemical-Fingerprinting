def fingerprintingWorkflow(IDLabel: list, 
                           algorithm: str, 
                           verbose: bool = True, 
                           parameters = None, 
                           algorithmParameters = None, 
                           sourceDirectory = None):
    
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

    #eof#
    
    Fingerprinting Workflow function takes mass spec features and identifies 
    the most diagnostic features that are predictive of a source!
    
    To use this function, place the x, y, and X_mixture data in a single
    location and insure the names for these files are as follows: 
            
            X.txt
            y.txt
            X_mixtures.txt

    Function Parameters Explanation:
        -ID Label: Sets list of IDs for the sources of the data. Provide a list
        of strings corrisponding to these source IDs:
            
            Example: ["AFFF","BL","LL", "WWTP"]
        
        -algorithm: Sets algorithm that will be used to process the data. 
        Provide the name of the algorithm to use from the following list:
            
            "SVC" = SVC algrithm
            
        -verbose: Determines whether or not the code will display informational
        outputs in the console as the script is running. Note: Disabling 
        outputs can speed up the runtime of the code.
            
            True = display outputs in the console.
            False = do not display outputs in the console.
    
        -parameters: Provide predetermined parameters for the testing metrics 
        of the script. Script can request for inputs, use default values, or 
        use provided values.
        (Note: To provide predetermined values, create a 
        fingerprintParametersClass object (from V2_Fingerprinting_functions
        script) to store the values in and pass that object into the script.)
            
            None = Ask the user to provide each parameter individually into 
            the console during initial execution of the code.
            
            "default" = Use default parameters (defined in the 
            generateTestingParameters function in the 
            V2_Fingerprinting_Functions script).
            
            fingerprintParametersClass: Script will test all provided 
            parameters to make sure inputs are valid.
        
        -algorithmParameters: Provide predetermined parameters for hyper 
        parameterization. These inputs are algorithm dependant. If using 
        predetermeined parameters, use corrisponding object class 
        for that algorithm. Enter "default" for default hyper parameters for 
        any algorithm, or enter None to manually enter in values in the 
        command prompt.
        
            "default": default parameters (defined in the 
            "generate[ALGORITHM]Parameters" function in the 
            V3_Fingerprinting_Functions script).
            
            None: Ask the user to provide each parameter individually into the 
            console during initial execution of the code. Input requested will 
            depend on algorithm. 
            
            *IF USING SVC ALGORITHM, USE a fingerprintSVCParametersClass:  
            Script will test all provided SVC parameters to make sure inputs 
            are valid.
        
        -sourceDirectory: Provide a location for the source data that will be 
        evaluated by this function.
            
        ***NOTE: PLEASE USE FORWARD SLASH /, NOT BACKSLASH \ FOR FILE PATHS or 
        else you may recieve an error***
            
            None = Ask user to select location of source data through explorer.
            
            "C:/ExamplePath" = Use defined path as source of user data.
    
    Examples of exectution: 
        
        fingerprintingWorkflow(["LL","JP"],"SVC",True,"Default", "default")
            Executes function using LL and JP as selected sourceIDs, SVC as 
            selected algorithm, verbose set to True (show outputs), use default
            values for parameters, use default hyper parameters for SVC 
            processing, and ask the user to select location of data.

        fingerprintingWorkflow(["AFFF"],"SVC")
            Executes function using AFFF as selected sourceID, SVC as selected 
            algorithm, verbose set to True by default (show outputs), ask user 
            to provide parameters, ask user to provide hyper parameters for SVC
            processing, and ask the user to select location of data.
        
        fingerprintingWorkflow(["WWTP","TTL","PPT"],"SVC",False, 
                               definedParametersVariable, 
                               definedSvcHyperParameters, 
                               "C:/ScienceRules/SourceData")
            Executes function using WWTP, TTL, and PPT as selected sourceIDs, 
            SVC as selected algorithm, verbose set to False (do not show 
            outputs), use user provide parameters (stored as a 
            fingerprintParametersClass), use user provide hyper parameters for 
            SVC (stored as a fingerprintSVCParametersClass), and use 
            C:/ScienceRules/SourceData as source files destination.

    """

    #Import modules
    import pandas as pd
    import numpy as np
    import os
    import sys
    from sklearn import preprocessing
    
    #Import fingerprint functions
    from V3_Fingerprinting_functions import selectFolder
    from V3_Fingerprinting_functions import testFingerprintFilePaths
    from V3_Fingerprinting_functions import generateTestingParameters
    from V3_Fingerprinting_functions import generateSVCParameters
    from V3_Fingerprinting_functions import importFingerprintData
    from V3_Fingerprinting_functions import misc_ml_storing_parameters
    from V3_Fingerprinting_functions import set_hyperparameter_domain
    from V3_Fingerprinting_functions import set_hyperparameter_value
    from V3_Fingerprinting_functions import Classifier
    from V3_Fingerprinting_functions import Classifier2
    from V3_Fingerprinting_functions import test_metric
    from V3_Fingerprinting_functions import model_printout
    from V3_Fingerprinting_functions import model_execute
    from V3_Fingerprinting_functions import model_execute2
    from V3_Fingerprinting_functions import model_results
    from V3_Fingerprinting_functions import plot_results
    from V3_Fingerprinting_functions import Norm_Importance
    from V3_Fingerprinting_functions import importance
    from V3_Fingerprinting_functions import saving_plotting_imp
    from V3_Fingerprinting_functions import retuning
    from V3_Fingerprinting_functions import retuning2
    from V3_Fingerprinting_functions import diagnostic_rerun
    from V3_Fingerprinting_functions import final_plots
    from V3_Fingerprinting_functions import fingerprintParametersClass
    from V3_Fingerprinting_functions import storingParametersClass
    from V3_Fingerprinting_functions import modeledResultsClass
    from V3_Fingerprinting_functions import importanceMetricsClass
    from V3_Fingerprinting_functions import createFolderPath
    
    #Have user select folder if sourceDirectory is not provided
    if(sourceDirectory == None):
        #Set working directory for sample files
        print("\nPlease select directory containing sample data.")
        sourceDirectory = selectFolder()
        os.chdir(sourceDirectory)
    #If sourceDirectory provided, check if path actually exists.
    else:
        #Confirm supplied directory is valid
        pathTest = os.path.isdir(sourceDirectory)
        #If path does not exist, raise exception and exit script.
        if(pathTest == False):
            raise ValueError('Error: "'+str(sourceDirectory) +'" is not a valid location. Please provide valid location and rerun script.')
        else:
            os.chdir(sourceDirectory)
            
    #Test if sourceDirectory contains required data for processing.
    testFingerprintFilePaths(sourceDirectory)
        
    #Import data
    X, y, X_mixtures, featureIndex = importFingerprintData()
    #y consists of multiple sources. These sources need to be converted to dummy variables that indicate the presence/absence of a source in a sample
    dummy = pd.get_dummies(y)
    
    #Test if provided parameters are valid. Set parameters to default if "default" is provided.
    parameters = generateTestingParameters(parameters, verbose)
    
    #Test if provide algorithm is valid.
    validAlgorithm = ["SVC"]
    if not algorithm in validAlgorithm:
        raise ValueError('Error: Invalid algorithm selected: " ' + algorithm + '". Please enter one of the following algorithms when rerunning the script: \n' + 
                 '\nSVC')
    else:
        pass
    
    folderName = "Final Results " + algorithm
    createFolderPath(sourceDirectory, folderName, verbose)
    #Set starting SourceID value for itteration through ID labels.
    SourceID = 0
    
    if(algorithm == 'SVC'):
        hyperParameterConstraints = generateSVCParameters(algorithmParameters)
    #If not, raise exception
    else:
        raise ValueError('Error: Invalid algorithm selected: " ' + algorithm + '". Please enter one of the following algorithms when rerunning the script: \n' + 
                         '\nSVC')
    #Set Hyper Parameter Domain
    hyperparameter_domain = set_hyperparameter_domain(algorithm, hyperParameterConstraints)

    #Run function for each provided Source ID using defined parameters:
    for IDName in IDLabel:
        # Confirm chosen classification tool is valid

      
        #Identify source would you like to examine (i.e., 0-n"). Not currently in use.
        #WW = 0
        #Pristine = 1
        #Ag = 2
        #SW = 3
        #Manure = 4
        
        #Set y dataset using sourceID
        y_dummy = dummy.iloc[:,SourceID]
        #Print dummy information if verbose is on.
        if (verbose == True):
            print("\nSource Name:" + IDName)
            print('\nDummies:\n', dummy)
            print('y', y_dummy)
        else:
            pass
        
        #Define test size based on test_size parameter.
        test_size_tuning = int(round(len(y_dummy)*parameters.test_size,0))  
    
        #Generate empty numpy array for modeling, currently array not in use.
        storingParameters = misc_ml_storing_parameters(algorithm, parameters.n_combos)
        
        #Show header if verbose set to True.
        if(verbose == True):
            print('Model tuning')
        else:
            pass
        
        #Select hyper parameters, performs train test split, calculated balanced accuracy, and generates ML model outputs.
        #modelExecutedResults = model_execute(algorithm, X, y_dummy, X_mixtures, parameters.n_combos, parameters.n_rs, test_size_tuning, hyperParameterConstraints)
        modelExecutedResults = model_execute2(algorithm, X, y_dummy, X_mixtures, parameters.n_combos, parameters.n_rs, test_size_tuning, hyperParameterConstraints, hyperparameter_domain, verbose)
        #raise ValueError("Stop test") 
        #Rearrange scores, removes nan values, sorts the top scores, and save the output.
        modeledResults = model_results(algorithm, modelExecutedResults.results, verbose)
    
        #Generate plots for modeled data.
        plot_results('Tuning', algorithm, modeledResults, IDName, storingParameters.classifier)
        return
        #return modeledResults,modelExecutedResults
        #raise ValueError("Stop test")    
        #Show header if verbose is set to True.
        if(verbose == True):
            print('Classifier Coefficent / Importance determination')
        else:
            pass
        #Calculate importance of each feature.
        normImpMetrics = importance(algorithm, X, y_dummy, X_mixtures, parameters.test_size, parameters.importance_iterations, modeledResults.hyperParameterTop)
        ForSorting, ForSortingMix = saving_plotting_imp(algorithm, IDName, X, X_mixtures, normImpMetrics, featureIndex)
    
        #Show header if verbose set to True.
        if(verbose == True):
            print('Retuning')
        else:
            pass
        
        #TEST RETUNE
        #Rank sort the X (i.e., chemical data) based on the ranked importance
        X_Retuning = ForSorting[ :, ForSorting[1].argsort()[::-1]]
        #Rank sort the X_mixtures (i.e., chemical data) based on the ranked importance              
        X_mixtures_Retuning = ForSortingMix[ :, ForSortingMix[1].argsort()[::-1]]   
        #Retain only the n most important features from the rank sorted X data
        X_Retuning = X_Retuning[2:,0:parameters.num_features]
        #Retain only the n most important features from the rank sorted X_mixtures data                        
        X_mixtures_Retuning = X_mixtures_Retuning[2:,0:parameters.num_features]                          
    
        #transforming the test size into an integer for train-test split
        test_size_retuning = int(round(len(y)*parameters.test_size,0))
         
        #Retune the model with only the n most important features   
        #retunedResults = retuning2(algorithm, y_dummy, parameters.test_size, parameters.n_combos, parameters.n_rs, parameters.num_features, ForSorting, ForSortingMix, modeledResults.hyperParameterTop, hyperParameterConstraints, hyperparameter_domain)
        #TESTING
        retunedResults = model_execute2(algorithm, X_Retuning, y_dummy, X_mixtures_Retuning, parameters.n_combos, parameters.n_rs, test_size_retuning, hyperParameterConstraints, hyperparameter_domain, verbose)
#(algorithm, X_Retuning, y_dummy, X_mixtures_Retuning, parameters.n_combos, parameters.n_rs, test_size_retuning, hyperParameterConstraints, hyperparameter_domain, verbose)"
        #Rearrange scores, removes nan values, sorts the top scores, and save the output for retuned results.
        retunedModeledResults = model_results(algorithm, retunedResults.results, verbose)
        #Generate plots for retuned modeled results.
        plot_results('Retuning', algorithm, retunedModeledResults, IDName, storingParameters.classifier)
    
        #Rerunning
        #TESTING
        #mean_pred_all, mean_proba_known, test_actual, test_pred = diagnostic_rerun(algorithm, IDName, X_Retuning, y_dummy, X_mixtures_Retuning, parameters.test_size, parameters.final_iterations, modeledResults.hyperParameterTop)
        mean_pred_all, mean_proba_known, test_actual, test_pred = diagnostic_rerun(algorithm, IDName, X_Retuning, y_dummy, X_mixtures_Retuning, parameters.test_size, parameters.final_iterations, retunedModeledResults.hyperParameterTop)
        final_plots(y_dummy, IDName, mean_proba_known, test_actual, test_pred)
        
        #Itterate ID Source
        SourceID += 1