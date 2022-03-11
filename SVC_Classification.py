def SVC_Classification(IDLabel, algorithm, verbose, parameters, algorithmParameters, X, X_mixtures, y, dummy, featureIndex, sourceDirectory):
    from V4_Fingerprinting_functions import set_hyperparameter_domain
    from V4_Fingerprinting_functions import set_hyperparameter_value
    from V4_Fingerprinting_functions import Classifier
    from V4_Fingerprinting_functions import Classifier2
    from V4_Fingerprinting_functions import misc_ml_storing_parameters
    from V4_Fingerprinting_functions import model_execute2
    from V4_Fingerprinting_functions import model_results
    from V4_Fingerprinting_functions import plot_results
    from V4_Fingerprinting_functions import importance
    from V4_Fingerprinting_functions import saving_plotting_imp
    from V4_Fingerprinting_functions import diagnostic_rerun
    from V4_Fingerprinting_functions import final_plots
    from V4_Fingerprinting_functions import generateSVCParameters

    SourceID = 0
    hyperParameterConstraints = generateSVCParameters(algorithmParameters)
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