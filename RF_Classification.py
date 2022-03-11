class fingerprintRFParametersClass:
    def __init__(self, setup):

        #Sets Default testing parameters for the fingerprint function.
        if(type(setup) == str):
            setup = setup.lower()
        else:
            pass
        if((setup == "default") or (setup == "None")):
            self.n_estimators = 100
            self.criterion = 'gini'
            self.max_depth = None
            self.min_samples_split = 2
            self.min_samples_leaf = 1
            self.min_weight_fraction_leaf = 0.0
            self.max_features = 'auto'
            self.max_samples = None

        elif(setup == 'ask'):
            self.n_estimators = "undefined"
            self.criterion = "undefined"
            self.max_depth = "undefined"
            self.min_samples_split = "undefined"
            self.min_samples_leaf = "undefined"
            self.min_weight_fraction_leaf = "undefined"
            self.max_features = "undefined"
            self.max_samples = "undefined"
            #Define number of trees in the forest  (Reccomend 100-1000).
            while(self.n_estimators == "undefined"):
                try:
                    self.n_estimators = int(input("Define number of trees in the forest  (Reccomend 100-1000): "))
                except:
                    print("\nError: Invalid entry. Please enter an integer value.")
                    self.n_estimators = "undefined"
            #Define the function to measure the quality of a split.
            while(self.criterion == "undefined"):
                try:
                    self.criterion = str(input('Define the function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.  ("gini", "entropy"): '))
                    self.criterion.replace('"','')
                    if ((self.criterion == "gini") or (self.criterion == "entropy")):
                        pass
                    else:
                        print('\nError: Invalid entry. Please enter either "gini" or "entropy".')
                        self.criterion = "undefined"
                except:
                    print("\nError: Invalid entry. Please enter an integer value.")
                    self.criterion = "undefined"
            #Define the maximum depth of the tree.
            while(self.max_depth == "undefined"):
                try:
                    self.max_depth = (input("Define the maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples: "))
                    self.max_depth = self.max_depth.replace('"','')
                    if ((self.max_depth == "None") or (self.max_depth == "none")):
                        self.max_depth = None
                    else:
                        self.max_depth = int(self.max_depth)
                except:
                    print("\nError: Invalid entry. Please enter an integer value.")
                    self.max_depth = "undefined"
            #Define minimum number of samples required to split an internal node (Reccomend 1-5).
            while(self.min_samples_split == "undefined"):
                try:
                    self.min_samples_split = int(input("Define minimum number of samples required to split an internal node (Reccomend 1-5): "))
                except:
                    print("\nError: invalid entry. Please enter an interger value.")
                    self.min_samples_split = "undefined"
            #Define minimum number of samples required to be at a leaf node (Recommended to keep low, 1-10")
            while(self.min_samples_leaf == "undefined"):
                try:
                    self.min_samples_leaf = int(input("Define minimum number of samples required to be at a leaf node (Recommended to keep low, 1-10): "))
                except:
                    print("\nError: invalid entry. Please enter an integer value.")
                    self.min_samples_leaf = "undefined"
            #Define minimum weighted fraction of the sum total of weights of all the input samples (Between 0 and 1).
            while(self.min_weight_fraction_leaf == "undefined"):
                try:
                    self.min_weight_fraction_leaf = float(input("Define minimum weighted fraction of the sum total of weights of all the input samples (Between 0 and 1): "))
                    if(self.min_weight_fraction_leaf >= 1):
                        print("\nError: invalid entry. Please enter a decimal value less than 1.")
                        self.min_weight_fraction_leaf = "undefined"
                except:
                    print("\nError: invalid entry. Please enter a decimal value less than 1.")
                    self.min_weight_fraction_leaf = "undefined"
            #Define the number of features to consider when looking for the best split
            while(self.max_features == "undefined"):
                try:
                    self.max_features = (input('Define the number of features to consider when looking for the best split ("sqrt", "log2", None, float): '))
                    self.max_features = self.max_features.replace('"','')

                    if ((self.max_features == "sqrt") or (self.max_features == "log2")): 
                        pass
                    elif ((self.max_features == "None") or (self.max_features == "none")):
                        self.max_features = None
                    else:
                        self.max_features = float(self.max_features)
                except:
                    print('except Error: invalid entry. Please enter either "sqrt", "log2", None, or a float value.')
                    self.max_features = "undefined"
            #Define the percentage of samples to draw from X to train each base estimator.
            while(self.max_samples == "undefined"):
                try:
                    self.max_samples = float(input("Define the percentage of samples to draw from X to train each base estimator for bootstrapping (Between 0 and 1): "))
                    if(self.max_samples >= 1):
                        print("\nError: invalid entry. Please enter a decimal value less than 1.")
                        self.max_samples = "undefined"
                except:
                    print("\nError: invalid entry. Please enter a decimal value less than 1.")
                    self.max_samples = "undefined"
        else:
            print("Invalid entry. Auto generating default parameters")
            self.n_estimators = 100
            self.criterion = 'gini'
            self.max_depth = None
            self.min_samples_split = 2
            self.min_samples_leaf = 1
            self.min_weight_fraction_leaf = 0.0
            self.max_features = 'auto'
            self.max_samples = None
    

def RF_Classification(IDLabel, algorithm, verbose, parameters, algorithmParameters, X, X_mixtures, y, dummy, featureIndex, sourceDirectory):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    
    if type(algorithmParameters) == fingerprintRFParametersClass:
        pass
    else:
        algorithmParameters = fingerprintRFParametersClass(algorithmParameters)
    SourceID = 0
    y_dummy = dummy.iloc[:,SourceID]
    test_size_tuning = int(round(len(y_dummy)*parameters.test_size,0))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_tuning,random_state=None,shuffle=True)
    model = RandomForestClassifier(algorithmParameters.n_estimators, 
                                   criterion = algorithmParameters.criterion,
                                   max_depth= algorithmParameters.max_depth, 
                                   min_samples_split = algorithmParameters.min_samples_split, 
                                   min_samples_leaf= algorithmParameters.min_samples_leaf,
                                   min_weight_fraction_leaf= algorithmParameters.min_weight_fraction_leaf, 
                                   max_features = algorithmParameters.max_features,
                                   max_samples = algorithmParameters.max_samples)
    model.fit(X_train, y_train)
    y_test_predict = model.predict(X_test)
    y_train_predict = model.predict(X_train)
    train_matrix = confusion_matrix(y_train, y_train_predict)
    test_matrix = confusion_matrix(y_test, y_test_predict)
    print(model.score(X_test, y_test))
    print("train conf matrix")
    print(train_matrix)
    print("test conf matrix")
    print(test_matrix)
    print("predict test")
    print(y_test_predict)
    print("predict train")
    print(y_train_predict)
    
    