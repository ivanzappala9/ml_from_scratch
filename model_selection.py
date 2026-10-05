import numpy as np

def stratified_train_test_split(X,y,test=0.20,random_state=None):

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    if y.size == 0:
        raise ValueError("The dataset must not be empty.")

    if not 0 < test < 1:
        raise ValueError("test must be between 0 and 1.")


    rng = np.random.default_rng(random_state)
    X=np.array(X)
    y=np.array(y)    
    n_data = len(y)   #Number of total examples
    unique, counts = np.unique(y,axis=0,return_counts=True) #unique contains the target classes and counts their absolute frequencies

    index_per_class=np.round(test*counts)   #Here we are calculating how many indexes per class we need so that out split is actually stratified
    indexes_unique=[]                       #This is going to be a list that will contain the indexes of our test set
    
    for i,val in enumerate(unique):         #Now we cicle for each target class
        tmp_vect=np.where(y==val)           #tmp_vect is an array and contains the indexes of y whose value is val
		
		#Then we extract a random subset (without replacement) of those indexes, with the correct size previously calculated, 
		#and we append those indexes in our array of indexes        
        indexes_unique.append(rng.choice(tmp_vect[0],size=int(index_per_class[i]),replace=False)) 

    #np.random.choice returns an array, so indexes_unique is an array of array, we need one unique array,
    #so we concatenate them
    test_indexes=np.concatenate(indexes_unique)
    train_indexes=np.setdiff1d(np.arange(n_data),test_indexes) #To obtain the train indexes we operate the insiemistic difference
                                                               #between the set of all indexes and the set of the test_indexes

    X_test=X[test_indexes]
    y_test=y[test_indexes]
    X_train=X[train_indexes]
    y_train=y[train_indexes]

    return X_train,X_test,y_train,y_test
