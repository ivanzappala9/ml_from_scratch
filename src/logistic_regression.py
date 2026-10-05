import numpy as np
from scipy.special import expit
from metrics import crossed_entropy, accuracy_score

class LogisticRegressionMiniBatch():
    
    #rate:       Learning rate of gradient descendent
    #rate_decay: Tells us if the learning rate will diminuish for each iteration. In concrete, is this is True
    #            the learning rate at the epoch n must be rate_n = rate_0 * (1/(1+n))
    #n_epochs:   
    #batch_tam:  The dimension of the subset of data to train the gradient descendent

    def __init__(self, rate=0.1, rate_decay=False, n_epochs=100,batch_tam=64,early_stopping=False, paciencia=3,random_state=None):
        self.rate=rate
        self.rate_decay=rate_decay
        self.batch_tam=batch_tam
        self.n_epochs=n_epochs
        self.random_state=random_state
        self.early_stopping = early_stopping
        self.paciencia=paciencia

        self.n_data=None
        self.n_features=None
        self.weights=None
        self.classes=[]
        self.loss_history = []
        self.val_loss_history = []
	
	
    def __sigmoide(self,x):
	    return expit(x)
	    

	
	#n_epochs: is the number of iteration of the descendent gradient
	#salida_epoch: If set to True, the model will print the cross-entropy loss and accuracy at the start
	#			   and after each training epoch — both for the training set and for the validation set.
	#early_stoppping:
	#paciencia:
    def fit(self, X, y, Xv=None, yv=None):
        X_no_uno=np.asarray(X,dtype=float)		#Initial dataset
        y=np.array(y)
        
        if X_no_uno.ndim != 2:
            raise ValueError(
                "X must be a 2D array with shape (n_samples, n_features)."
            )

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X_no_uno.shape[0] != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        if X_no_uno.shape[0] == 0 or X_no_uno.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature.")

        if not np.isfinite(X_no_uno).all():
            raise ValueError("X must contain only finite values.")

        if not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
            raise ValueError("y must contain both classes, encoded as 0 and 1.")

        # Controllo dei parametri di addestramento
        if not np.isfinite(self.rate) or self.rate <= 0:
            raise ValueError("rate must be a positive finite number.")

        if not isinstance(self.n_epochs, (int, np.integer)) or self.n_epochs <= 0:
            raise ValueError("n_epochs must be a positive integer.")

        if not isinstance(self.batch_tam, (int, np.integer)) or self.batch_tam <= 0:
            raise ValueError("batch_tam must be a positive integer.")
        
        self.loss_history = []
        self.val_loss_history = []
        self.n_data, self.n_features = X_no_uno.shape
        self.classes=list(np.unique(y))
        ones = np.ones((self.n_data, 1))	#Auxiliar array: contains a column of ones
        X = np.hstack((ones, X_no_uno))     #Full dataset: we have added a column of ones to make calculations after easier
        
        self.weights = np.zeros(self.n_features + 1)
        
        min_entropy=1e15 #This will contain the minimum of the entropy. It's initial value is a very high value (theoretically should be +inf)
        count_pac=0 #Counter for paciencia. When it reaches the max value the train stops
		
		#If we have also tha validation set, we modify it in order to have the column of ones
        if (Xv is not None) and (yv is not None):
            Xv = np.array(Xv)
            yv = np.array(yv)

        
        current_rate=self.rate
        rng = np.random.default_rng(self.random_state)
        #TRAINING CICLE (GRADIENT DESCENDENT)
        for epoch in range(self.n_epochs):
        
            if (self.rate_decay==True):
                current_rate=self.rate/(1+epoch)  #Modifying the learning rate
                
            
            
            shuffled_indexes = rng.permutation(self.n_data)
            
            for start in range(0,self.n_data, self.batch_tam):
                batch_indexes = shuffled_indexes[start:start+self.batch_tam]
			    #Batch subsets            
                y_batch=y[batch_indexes]
                X_batch=X[batch_indexes]
                
                #Weight update
                y_pred=self.__sigmoide(np.dot(X_batch,self.weights))  #Predicted values
                tmp1=y_batch-y_pred 
                gradient= np.dot(X_batch.T, tmp1) / len(batch_indexes)
                self.weights += current_rate*gradient	#Updating all the weight at the same time
             
                    
                
                
            if (self.early_stopping==True) and (Xv is not None) and (yv is not None):
                if count_pac==self.paciencia:
                    break
                    
                val_proba = self.predict_proba(Xv)[:, 1]
                train_loss=crossed_entropy(yv,val_proba)     #crossed_entropy of train set
                self.val_loss_history.append(train_loss)
                
                if train_loss > min_entropy:
                    count_pac+=1
                else:
                    min_entropy = train_loss
                    best_weights = self.weights.copy()
                    count_pac=0

                
            
            # Perdita sull'intero training set a fine epoca
            train_proba = self.__sigmoide(X @ self.weights)
            train_loss = crossed_entropy(y, train_proba)

            self.loss_history.append(train_loss)
            
        if self.early_stopping and best_weights is not None:
            self.weights = best_weights.copy()
        return self
    
    #This function receives a matrix (or an array) of examples to predict returning their probabilities of belonging to the positive class
    #======  NOTE: THIS FUNCTION RETURN A PROBABILITY FOR EACH EXAMPLE (a number between 0 and 1)=========
    def predict_proba(self, X_new):
        if self.weights is None:
            raise ValueError("The model is not trained yet.")
        X_new = np.array(X_new)
    
        if X_new.ndim != 2:
            raise ValueError(
                "X must be a 2D array with shape (n_samples, n_features)."
            )

        if X_new.shape[1] != self.n_features:
            raise ValueError(
                "X must have the same number of features as the training set."
            )

        if not np.isfinite(X_new).all():
            raise ValueError("X must contain only finite values.")
    
        #Adding the column of ones
        ones = np.ones((X_new.shape[0], 1))
        X_new = np.hstack((ones, X_new))
    
        #Calculating the probability for each example: If we have n we will have an array of of n probabilities
        proba_1 = self.__sigmoide(X_new @ self.weights)

        # Colonne: probabilità della classe 0 e della classe 1
        return np.column_stack((1 - proba_1, proba_1))

    #This function receives a matrix (or an array) of examples and predicts their class
    #======= NOTE: THIS FUNCTION RETURN THE CLASS ===========
    def predict(self,X_new):
        proba=self.predict_proba(X_new)
        return (proba[:, 1] >= 0.5).astype(int)




