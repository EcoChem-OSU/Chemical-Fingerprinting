def fingerprintingWorkflow(algorithm: str, IDLabel: list = ["sample"], verbose: bool = True, parameters = "default", algorithmParameters = "default", sourceDirectory = None):
    
    """
    Fingerprinting Workflow function takes mass spec features and identifies the most diagnostic features that are predictive of a source!
    
    To use this function, place the x, y, and X_mixture data in a single location and insure the names for these files are as follows: 
            
            X.txt
            y.txt
            X_mixtures.txt

    Function Parameters Explanation:
        ID Label: Sets list of IDs for the sources of the data. Provide a list of strings corrisponding to these source IDs:
            
            Example: ["AFFF","BL","LL", "WWTP"]
        
        algorithm: Sets algorithm that will be used to process the data. Provide the name of the algorithm to use from the following list:
            
            "SVC" = SVC algrithm
            
        verbose: Determines whether or not the code will display informational outputs in the console as the script is running. Note: Disabling outputs can speed up the runtime of the code.
            
            True = display outputs in the console.
            False = do not display outputs in the console.
    
        parameters: Provide predetermined parameters for the testing metrics of the script. Script can request for inputs, use default values, or use provided values.
        (Note: To provide predetermined values, create a fingerprintParametersClass object (from V2_Fingerprinting_functions script) to store the values in and pass that object into the script.)
            
            None = Ask the user to provide each parameter individually into the console during initial execution of the code.
            "default" = Use default parameters (defined in the generateTestingParameters function in the V2_Fingerprinting_Functions script).
            fingerprintParametersClass: Script will test all provided parameters to make sure inputs are valid.
        
        algorithmParameters: Provide predetermined parameters for hyper parameterization. These inputs are algorithm dependant. If using predetermeined parameters, use corrisponding object class 
        for that algorithm. Enter "default" for default hyper parameters for any algorithm, or enter None to manually enter in values in the command prompt.
        
            "default": default parameters (defined in the "generate[ALGORITHM]Parameters" function in the V3_Fingerprinting_Functions script).
            None: Ask the user to provide each parameter individually into the console during initial execution of the code. Input requested will depend on algorithm.
            *IF USING SVC ALGORITHM, USE: fingerprintSVCParametersClass:  Script will test all provided SVC parameters to make sure inputs are valid.
        
        sourceDirectory: Provide a location for the source data that will be evaluated by this function.
            ***NOTE: PLEASE USE FORWARD SLASH /, NOT BACKSLASH \ FOR FILE PATHS or else you may recieve an error***
            None = Ask user to select location of source data through explorer.
            "C:/ExamplePath" = Use defined path as source of user data.
    
    Examples of exectution: 
        
        fingerprintingWorkflow(["LL","JP"],"SVC",True,"Default", "default")
            Executes function using LL and JP as selected sourceIDs, SVC as selected algorithm, verbose set to True (show outputs), use default values for parameters, use default 
            hyper parameters for SVC processing, and ask the user to select location of data.

        fingerprintingWorkflow(["AFFF"],"SVC")
            Executes function using AFFF as selected sourceID, SVC as selected algorithm, verbose set to True by default (show outputs), ask user to provide parameters, 
            ask user to provide hyper parameters for SVC processing, and ask the user to select location of data.
        
        fingerprintingWorkflow(["WWTP","TTL","PPT"],"SVC",False,definedParametersVariable, definedSvcHyperParameters, "C:/ScienceRules/SourceData")
            Executes function using WWTP, TTL, and PPT as selected sourceIDs, SVC as selected algorithm, verbose set to False (do not show outputs), 
            use user provide parameters (stored as a fingerprintParametersClass), use user provide hyper parameters for SVC (stored as a fingerprintSVCParametersClass), 
            and use C:/ScienceRules/SourceData as source files destination.

    """
    """
    Created on Sat Dec  4 12:37:34 2021
    
    @author: Gerrad
    """
    #Import modules
    import pandas as pd
    import os
    
    #Import fingerprint functions
    from V4_2_SVC_Classification import SVC_Classification
    from RF_Classification import RF_Classification
    from V4_2_Fingerprinting_functions import selectFolder
    from V4_2_Fingerprinting_functions import testFingerprintFilePaths
    from V4_2_Fingerprinting_functions import importFingerprintData
    from V4_2_Fingerprinting_functions import fingerprintParametersClass
    from V4_2_Fingerprinting_functions import createFolderPath
    
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
    if("fingerprintParametersClass" in str(type(parameters))):
        parameters.validate()
    else:
        p = fingerprintParametersClass()

        if(parameters == "default" or parameters == "Default"):
            parameters = p
        else:
            p.generateParameters(parameters)
            parameters = p

    #Test if provide algorithm is valid.
    validAlgorithm = ["SVC","RF"]
    if not algorithm in validAlgorithm:
        raise ValueError('Error: Invalid algorithm selected: " ' + algorithm + '". Please enter one of the following algorithms when rerunning the script: \n' + 
                 '\nSVC')
    else:
        pass
    
    folderName = "Final Results " + algorithm
    createFolderPath(sourceDirectory, folderName, verbose)
    #Set starting SourceID value for itteration through ID labels.

    
    if(algorithm == 'SVC'):

        SVC_Classification(IDLabel, algorithm, verbose, parameters, algorithmParameters, X, X_mixtures, y, dummy, featureIndex, sourceDirectory)
    #If not, raise exception
    elif(algorithm == 'RF'):
        RF_Classification(IDLabel, algorithm, verbose, parameters, algorithmParameters, X, X_mixtures, y, dummy, featureIndex, sourceDirectory)
    else:
        raise ValueError('aError: Invalid algorithm selected: " ' + algorithm + '". Please enter one of the following algorithms when rerunning the script: \n' + 
                         '\nSVC')