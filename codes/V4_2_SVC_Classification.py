def SVC_Classification(IDLabel, algorithm, verbose, parameters, algorithmParameters, X, X_mixtures, y, dummy, featureIndex, sourceDirectory):
    from V4_2_SVC_Classification import _set_hyperparameter_domain
    from V4_2_SVC_Classification import _misc_ml_storing_parameters
    from V4_2_SVC_Classification import _model_execute
    from V4_2_SVC_Classification import _model_results
    from V4_2_SVC_Classification import _plot_results
    from V4_2_SVC_Classification import _importance
    from V4_2_SVC_Classification import _saving_plotting_imp
    from V4_2_SVC_Classification import _diagnostic_rerun
    from V4_2_SVC_Classification import _final_plots
    from V4_2_SVC_Classification import generateSVCParameters
    #import numpy as np

    SourceID = 0
    hyperParameterConstraints = generateSVCParameters(algorithmParameters)
    hyperparameter_domain = _set_hyperparameter_domain(algorithm, hyperParameterConstraints)

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
        #test_size_tuning = int(round(len(y_dummy)*parameters.test_size,0))  
        test_size_tuning = parameters.test_size  
    
    
        #Generate empty numpy array for modeling, currently array not in use.
        storingParameters = _misc_ml_storing_parameters(algorithm, parameters.n_combos)
        
        #Show header if verbose set to True.
        if(verbose == True):
            print('Model tuning')
        else:
            pass
        
        #Select hyper parameters, performs train test split, calculated balanced accuracy, and generates ML model outputs.
        #modelExecutedResults = model_execute(algorithm, X, y_dummy, X_mixtures, parameters.n_combos, parameters.n_rs, test_size_tuning, hyperParameterConstraints)
        modelExecutedResults = _model_execute(algorithm, X, y_dummy, X_mixtures, parameters.n_combos, parameters.n_rs, test_size_tuning, hyperParameterConstraints, hyperparameter_domain, verbose)
        #raise ValueError("Stop test") 
        #Rearrange scores, removes nan values, sorts the top scores, and save the output.
        modeledResults = _model_results(algorithm, modelExecutedResults.results, verbose)

        #Generate plots for modeled data.
        _plot_results('Tuning', algorithm, modeledResults, IDName, storingParameters.classifier)

        #return modeledResults,modelExecutedResults
        #raise ValueError("Stop test")    
        #Show header if verbose is set to True.
        if(verbose == True):
            print('Classifier Coefficent / Importance determination')
        else:
            pass
        #Calculate importance of each feature.
        normImpMetrics = _importance(algorithm, X, y_dummy, X_mixtures, parameters.test_size, parameters.importance_iterations, modeledResults.hyperParameterTop)
        ForSorting, ForSortingMix = _saving_plotting_imp(algorithm, IDName, X, X_mixtures, normImpMetrics, featureIndex)
    
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
        #test_size_retuning = int(round(len(y)*parameters.test_size,0))
        test_size_retuning = parameters.test_size
         
         
        #Retune the model with only the n most important features   
        #retunedResults = retuning2(algorithm, y_dummy, parameters.test_size, parameters.n_combos, parameters.n_rs, parameters.num_features, ForSorting, ForSortingMix, modeledResults.hyperParameterTop, hyperParameterConstraints, hyperparameter_domain)
        #TESTING
        retunedResults = _model_execute(algorithm, X_Retuning, y_dummy, X_mixtures_Retuning, parameters.n_combos, parameters.n_rs, test_size_retuning, hyperParameterConstraints, hyperparameter_domain, verbose)
        #(algorithm, X_Retuning, y_dummy, X_mixtures_Retuning, parameters.n_combos, parameters.n_rs, test_size_retuning, hyperParameterConstraints, hyperparameter_domain, verbose)"
        #Rearrange scores, removes nan values, sorts the top scores, and save the output for retuned results.
        retunedModeledResults = _model_results(algorithm, retunedResults.results, verbose)
        #Generate plots for retuned modeled results.
        _plot_results('Retuning', algorithm, retunedModeledResults, IDName, storingParameters.classifier)
    
        #Rerunning
        #TESTING
        #mean_pred_all, mean_proba_known, test_actual, test_pred = diagnostic_rerun(algorithm, IDName, X_Retuning, y_dummy, X_mixtures_Retuning, parameters.test_size, parameters.final_iterations, modeledResults.hyperParameterTop)
        mean_pred_all, mean_proba_known, test_actual, test_pred = _diagnostic_rerun(algorithm, IDName, X_Retuning, y_dummy, X_mixtures_Retuning, parameters.test_size, parameters.final_iterations, retunedModeledResults.hyperParameterTop)
        _final_plots(y_dummy, IDName, mean_proba_known, test_actual, test_pred)
        
        #Itterate ID Source
        SourceID += 1
        
class storingParametersClass():
    classifier = None
    results = None

class fingerprintSVCParametersClass():

    #Initial setup
    def __init__(self):      
            self.c_lower = 0.00001
            self.c_upper = 10
            self.c_divisions = 100000000
            
    def generateSVCParameters(self, parametersSVC = None, verbose = True):
        if(type(parametersSVC) == str):
            parametersSVC = parametersSVC.lower()
        else:
            pass
        if(parametersSVC == "default"):
            #Classifier tuning parameter lower limit.
            self.c_lower = 0.00001
            #Classifier tuning parameter upper limit.
            self.c_upper = 10
            #Classifier tuning parameter number of divisions within range.
            self.c_divisions = 100000000
        elif(parametersSVC == None):
            parametersSVC = fingerprintSVCParametersClass()
            while(self.c_lower == None):
                try:
                    self.c_lower = float(input("Define lower limit of C (SVC classifier tuning parameter) "))
                except:
                    print("\nError: invalid entry. Please enter a float value.")
                    self.c_lower = None
            while(self.c_upper == None):
                try:
                    self.c_upper = float(input("Define upper limit of C (SVC classifier tuning parameter) "))
                except:
                    print("\nError: invalid entry. Please enter a float value.")
                    self.c_upper = None
            #Not required?
            #while(self.c_divisions == None):
                #try:
                    #self.c_divisions = int(input("Define number of divisions in range for SVC classifier tuning parameter: "))
                #except:
                    #print("\nError: invalid entry. Please enter an integer value.")
                    #self.c_divisions = None

        else:
            raise ValueError('Error! Value "'+ parametersSVC + '" is not valid. Please create a fingerprintSVCParametersClass to pass into the function.')
        
    def validateParameters(self):
            #if(parametersSVC == fingerprintSVCParametersClass):
            try:
                self.c_lower = float(self.c_lower)
            except:
                raise ValueError("\nError: invalid entry. Please enter a float value.")
            try:
                self.c_upper = float(self.c_upper)
            except:
                raise ValueError("\nError: invalid entry. Please enter a float value.")

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

#Create arrays needed to run and store information for each ML model
def _misc_ml_storing_parameters(algorithm, n_combos):
    import numpy as np
    if algorithm == "SVC":
        sParameters = storingParametersClass()
        sParameters.classifier = 2
        sParameters.results = np.zeros((sParameters.classifier,n_combos))
        return (sParameters)

    else:
        print('whoops...undefined algorithm selected')

#set the domain of relevant tuning parameters which will be randomly drawn from
def _set_hyperparameter_domain(algorithm, hyperParameterConstraints):
    import numpy as np
    if algorithm == "SVC":
        #c=tuning parameter in Classifier. The first number is the lower limit, the second number is the upper limit, and the third number is the number of divisions within this range.
        #C_ = np.geomspace(0.000001,1000,num=100000,endpoint=True)
        #C_ = np.geomspace(parameters.c_lower,parameters.c_upper,parameters.c_divisions,endpoint=True)
        C_ = np.geomspace(hyperParameterConstraints.c_lower, hyperParameterConstraints.c_upper, num=100000000, endpoint=True)
        
        return(C_)

    else:
        print('whoops...undefined algorithm selected')

#select a hyperparameter tuning value
def _set_hyperparameter_value(algorithm, hyperparameter_domain):
    import random
    if algorithm == "SVC":
        C_ = hyperparameter_domain[:]
        C = random.choice(C_)  
        return(C)
    else:
        print('whoops...undefined algorithm selected')

#define the classifier to be used in the workflow 
def _Classifier(algorithm, hyperparameter_value, X_train, X_test, y_train, y_test, X, X_mixtures):
    from sklearn.metrics import confusion_matrix
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
def _test_metric(TrainAccuracy,TestAccuracy):
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
def _model_printout(algorithm, hyperparameter_value, avg_score):
    import numpy as np
    if algorithm == "SVC":
        print("     C = ", "{:.1e}".format(hyperparameter_value))
        print("     Averaged Balanced Accuracy of the Testing data = ", np.round(avg_score,2))       
        
        

#TESTING
#execute the model. This script selects hyper parameters, performs train test split, calculated balanced accuracy, and generates model outputs to be used later 
def _model_execute(algorithm, X, y, X_mixtures, n_combos, n_rs, test_size_tuning, hyperParameterConstraints, hyperparameter_domain, verbose):
    import numpy as np
    import timeit
    from V4_1_Fingerprinting_functions import train_test_split_grouped
    from V4_2_SVC_Classification import _set_hyperparameter_value
    
    i = 0 
    results = _misc_ml_storing_parameters(algorithm, n_combos)
    
    #Iterate n times
    while i < n_combos:                                                                         
    #Classifier requires tuning of different parameters.
  
        #select tuning parameter value
        hyperparameter_value = _set_hyperparameter_value(algorithm, hyperparameter_domain)
    
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
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_tuning,random_state=None,shuffle=True)
            X_train, X_test, y_train, y_test = train_test_split_grouped(X, y, test_size_tuning)
            #print("X_train: ", X_train, "\nX_test: ",X_test,"\ny_train: ",y_train,"\ny_test: ", y_test)
            #The Classifier fits the training and testing datastudent be supervised by a 
            y_test_pred, TrainAccuracy, TestAccuracy, coef_, proba_, Mix_proba_, pred_all = _Classifier(algorithm, hyperparameter_value, X_train, X_test, y_train, y_test, X, X_mixtures)
            #Apply metric score to evaluate overfitting
            metric_score[u] = _test_metric(TrainAccuracy,TestAccuracy)
    
        #except:
            #raise
            #Especially for sources with few samples, there is a chance that the training dataset has only one group (only absences are present). This will throw an error. THe try/except statment was included to prevent the script from stopping.
            #print('     Error: Rerunning...')
            
        #Calculate the average metric score...sometimes, the metric score is nan (unsure why), so nanmean is used to ignore these "values" 
        avg_score = np.nanmean(metric_score)
        #Store average metric score and corresponding params
        results.results[:,i] = np.array([avg_score,hyperparameter_value])
        if(verbose == True):
            _model_printout(algorithm, hyperparameter_value, avg_score)
        else:
            pass
        stop = timeit.default_timer() 
        if(verbose == True):
            print("     Time = ", np.round(stop - start, decimals = 0), "seconds")
        else:
            pass
        i = i+1
           
    return (results)

class modeledResultsClass():
        resultsSorted = None
        label = None
        hyperParameterTop = None
        bestParamsTop = None
        
#this script rearranges the scores, removes nan values, sorts the top scores, and saves the output.
def _model_results(algorithm, results, verbose):
    
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
def _plot_results(tuning, algorithm, modeledResults, IDName, n_params_Classifier):
    import matplotlib.pyplot as plt
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


class importanceMetricsClass():
    impMean = None
    impStd = None
    absMean = None
    importance = None

#Normalized Importance Function. This function takes the output of Classifier_CV_Imp and normalizes it for later use
def _Norm__importance(imp):
    import numpy as np
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
def _importance(algorithm, X, y, X_mixtures, test_size, importance_iterations, hyperparameter_top):
    import numpy as np
    import timeit
    from V4_1_Fingerprinting_functions import train_test_split_grouped
    from V4_2_SVC_Classification import _Classifier
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
                
                #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)
                X_train, X_test, y_train, y_test = train_test_split_grouped(X, y, test_size)
                #execute the Classifier_CV_Imp function                              
                #estimator = _Classifier
                y_test_pred, TrainAccuracy, TestAccuracy, coef_, proba_, Mix_proba_, pred_all = _Classifier(algorithm, hyperparameter_top, X_train, X_test, y_train, y_test, X, X_mixtures)

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
    normImpMetrics = _Norm__importance(imp)
    normImpMetrics.importance = imp
    all_stop = timeit.default_timer()
    print('Total Time: ', np.round((all_stop - all_start)/60, decimals = 0), "Minutes")
    return (normImpMetrics)

#saving and plotting
def _saving_plotting_imp(algorithm, IDName, X, X_mixtures, importanceMetrics,FeatureIndex):
    import numpy as np
    import matplotlib.pyplot as plt
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


#rerunning with n diagnostic features
def _diagnostic_rerun(algorithm, IDName, X, y, X_mixtures, test_size, final_iterations, hyperparameter_top):
    
    import math
    import numpy as np
    import timeit
    from V4_1_Fingerprinting_functions import train_test_split_grouped
    from V4_2_SVC_Classification import _Classifier
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
        #try:
        if algorithm == "SVC":
            
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)
            X_train, X_test, y_train, y_test = train_test_split_grouped(X, y, test_size, test_size_retuning)
            #execute the Classifier_CV_Imp function                              
            y_test_pred, TrainAccuracy, TestAccuracy, coef_, proba_, Mix_proba_, pred_all = _Classifier(algorithm, hyperparameter_top, X_train, X_test, y_train, y_test, X, X_mixtures)

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
        #except:
            #print('Error: Rerunning...')
    
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
def _final_plots(y, IDName, mean_proba_known, test_actual, test_pred):
    import scikitplot as skplt
    import numpy as np
    import matplotlib.pyplot as plt
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