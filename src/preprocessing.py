import numpy as np

class StandardNormalizer():
    """
    In questa classe viene implementata una normalizzazione normale del dataset,
    dunque ricevuto in ingresso un dataset X di (righe.colonne) = (samples,features),
    Viene normalizzata ogni colonna secondo la seguente formula: 
    Sia m_i la media della colonna i e d_i la sua deviazione standard, allora
    detta C_i la colonna i si ha C_i = (C_i - m_i)/d_i per i=1,...,n_colonne
    """
    def __init__(self):
        self.mean = None
        self.dev_std = None
        self.n_data = None
        self.n_features = None
    
    def fit(self,X):
        X=np.asarray(X, dtype=float)
        
        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array with shape (n_samples, n_features)."
            )
        
        self.n_data, self.n_features = X.shape

        self.mean = np.mean(X, axis=0)
        self.dev_std = np.std(X, axis=0)
        self.dev_std[self.dev_std == 0] = 1.0
        return self
    
    def transform(self,X):
        if self.mean is None:
            raise ValueError("You have to call fit() method before transform().")
            
        X=np.asarray(X, dtype=float)
        
        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array with shape (n_samples, n_features)."
            )

        if X.shape[1] != self.n_features:
            raise ValueError(
                "X must have the same number of features of the training set"
            )

        X=(X-self.mean)/self.dev_std
        return X
    
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)



class OneHotEncoder():
    """
    Questa classe implementa il one hot encoding in un generico dataset: Dunque
    vengono salvati colonna per colonna tutti i possibili valori che assume una features
    e ogni colonna si trasorma in N_unique_val colonne, dove N_unique_val è il numero di valori
    unici assunti da una certa feature. Esempio si ha una feature colore che assume i valori
    rosso, verde, blu; nella versione one-hot-encoded di tale feature colore diventa ora tre
    colonne: colore_rosso, colore_verde, colore_blu.
    """
    def __init__(self):
        self.categories = None
        self.n_features = None
        self.n_data = None
        
    def fit(self,X):
        self.categories = []
        X=np.array(X,dtype=object)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array with shape (n_samples, n_features)."
            )
            
        self.n_data, self.n_features = X.shape


        for idx_col in range(self.n_features):
            self.categories.append(np.unique(X[:,idx_col]))


        return self
    def transform(self,X):
    
        if self.n_features is None:
            raise ValueError("You have to call fit() method before transform().")
        
        X=np.array(X,dtype=object)               
        
        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array with shape (n_samples, n_features)."
            ) 
            
        if X.shape[1] != self.n_features:
            raise ValueError(
                "X must have the same number of features of the training set"
            )

        n_row = X.shape[0]    
        encoded_columns=[]  #List that will containt the encoded version of each column
        for idx_col in range(self.n_features):
            col_values = X[:,idx_col]
            unique_values = self.categories[idx_col]
            
            # Check that every value belongs to the learned categories.
            known_values = np.isin(col_values, unique_values)

            if not known_values.all():
                raise ValueError(
                    f"Unknown categories found in column {idx_col}."
                )
            
            one_hot_col = np.zeros((n_row,len(unique_values)))   #1. Creating a matrix of zeros with n_row:rows and (#_of_unique_values):columns
            
            for i,val in enumerate(unique_values):
                one_hot_col[:,i] = (col_values==val).astype(int)  #If the values of the column is equal to val we place 1, otherwise 0.
                                                                    #We do this for each val in unique_values

            encoded_columns.append(one_hot_col)
        
        one_hot_final=np.hstack(encoded_columns)
    
        return one_hot_final
            
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
        
        
        
        

