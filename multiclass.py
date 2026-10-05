import numpy as np
from logistic_regression import LogisticRegressionMiniBatch

class LogisticRegressionOVR():
    def __init__(self, rate=0.1, rate_decay=False, batch_tam = 64, n_epochs = 100, random_state=None):
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.n_epochs = n_epochs
        self.random_state = random_state

        self.classes = None
        self.models = dict()
        
    def fit(self, X, y):
        X=np.asarray(X,dtype=float)
        y=np.asarray(y)
        
        
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] != len(y):
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(
                "X must contain at least one sample and one feature."
            )
        
        
        self.classes = np.unique(y)
        
        if len(self.classes) < 2:
            raise ValueError("y must contain at least two classes.")
            
        self.n_features = X.shape[1]
        self.models = {}
        
        for c in self.classes:
            #Let's create a new target vector where entries are 1 if the class in y matches c, and 0 otherwise
            y_bin = (y == c).astype(int)
            #Crating the binary model to train for the class c
            lr_bin = LogisticRegressionMiniBatch(
                          rate = self.rate, 
                          rate_decay = self.rate_decay, 
                          batch_tam = self.batch_tam,
                          n_epochs = self.n_epochs,
                          random_state = self.random_state,
                          )
            #Fitting the model
            lr_bin.fit(X,y_bin)
            #Saving the model
            self.models[c] = lr_bin

        return self

    # This function receives in input a dataset and gives back a probability matrix where a given rows containes the probability of belonging to each class
    # So if we have 10 examples and 4 classes this functio will return a matrix (10 x 4) and the entry (5,2) containes the probability for the fifth examples of
    # belonging to the second class
    def _predict_scores(self, X):
        if self.classes is None or len(self.models) != len(self.classes):
            raise ValueError("The model is not trained yet.")
            
        scores = []  # This list will contain the probabilities of belonging to each class for all samples.
                     # For example, [0.1, 0.03, 0.4] could represent the probabilities of belonging to class 0,
                     # class 1, and class 2, respectively, for a single sample.

        for c in self.classes:
            proba_c = self.models[c].predict_proba(X)[:,1]   # Probability of each sample belonging to class c (versus all other classes)
            scores.append(proba_c)            # Reshape to ensure proba_c is a 1D array of shape (n_samples,)


        # At the end of the loop, probas will contain 'n_classes' arrays each of shape (n_samples,).
        # So probas looks like: [array(n_samples,), array(n_samples,), ..., array(n_samples,)]

        return np.column_stack(scores)

    
    def predict_proba(self, X):
        scores = self._predict_scores(X)
        row_sums = scores.sum(axis=1, keepdims=True)

        if not np.isfinite(scores).all():
            raise ValueError("The model produced non-finite scores.")

        if np.any(row_sums == 0):
            raise ValueError(
                "Cannot normalize scores: a row contains only zeros."
            )

        return scores / row_sums    
    
    # This function receives in input a dataset and gives back an array where the i-th entry contains the predicted class for the i-th example
    def predict(self, X):
        scores = self._predict_scores(X)  #Calculating the probability matrix
        index_max = np.argmax(scores, axis = 1)  #For each row we select the highest value, which means the highest probability
        # index_max is an array containing the indexes of the predicted class for each examples ---> Ex: [0,1,2,1,0,1,1,0,0,2,1]
        
        return self.classes[index_max]


