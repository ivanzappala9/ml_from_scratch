import numpy as np
from collections import Counter

class NaiveBayes:

    #Costruttore
    def __init__(self,alpha=1):
        self.alpha=alpha
        
        self.n_data = None
        self.n_features = None
        self.classes = None
        self.n_classes = None

        self.prior_prob = None
        self.feature_categories = {}
        self.feature_probs = {}
    
    #Metodo principale: lo usiamo per addestrare il modello. 
    #Accetta due array-like X e y, dove X è il nostro dataset categorico, dunque accetta sia stringhe che numeri,
    #y è il vettore che contiene le classi per ogni esempio
    def fit(self,X,y):

        #Qui prendiamo le dimensioni della matrice X e le classi da predirre
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        
        # Controllo delle dimensioni
        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array with shape (n_samples, n_features)."
            )

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] != len(y):
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        
        self.n_data, self.n_features = X.shape
        self.classes, class_counts =np.unique(y,return_counts=True) #vettore che contiene le diverse classi C_k
        self.n_classes = len(self.classes)
        
        
        #Step 1: Evaluating a-priori probability
        self.prior_prob=self.prior_prob = class_counts / self.n_data
        
        #Feature categoies è un dizionario: le keys sono l'indice della feature ed i valori sono un array contente
        #i valori unici di ciascuna feature
        self.feature_categories = {}
        

        self.feature_probs={}  #È il nostro dizionario di dizionari che conterrà le probabilità condizionate. È strutturato cosi: 
        """feature_probs{Classe_1:{ feat_1: {categoria_feat1_1: prob_1
                                             categoria_feat1_i: prob_i
                                            }
                                    feat_2: {categoria_feat2_1:prob
                                             categoria_feat2_2:prob ....
                                            }
                                  }
                         Classe_2:{ feat_1:{
                                            ...
                                           }
                                    feat_2:{
                                            ...
                                           }
                                 }
                        }
        """
        
        for feature_idx in range(self.n_features):          #Quello che facciamo qui è estrarre per ogni features i suoi valori unici
            self.feature_categories[feature_idx] = np.unique(X[:, feature_idx])

        
        for (class_idx, current_class) in enumerate(self.classes):  #Per ogni classe c calcoliamo le probabilita condizionate
            self.feature_probs[current_class]={}


            #Consideriamo la sottomatrice che contiene la classe che stiamo esaminando
            X_class = X[y==current_class]
            n_class = class_counts[class_idx]


            # Per ogni feature, calcoliamo P(X_i = value | C=classe)
            for feature_idx in range(self.n_features):
                category_probs = {}

                

                categories = self.feature_categories[feature_idx]
                n_categories = len(categories)

                counts = Counter(X_class[:, feature_idx])
                for category in categories:
                    count = counts.get(category, 0)
                    probability = (
                        (count + self.alpha)
                        / (n_class + self.alpha * n_categories)
                    )
                    category_probs[category] = probability
                    
                self.feature_probs[current_class][feature_idx] = (
                    category_probs
                )
        return self


    def _joint_log_proba(self, X):
        if self.classes is None:
            raise ValueError("The model is not trained yet.")

        X = np.asarray(X, dtype=object)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array with shape (n_samples, n_features)."
            )

        if X.shape[1] != self.n_features:
            raise ValueError(
                "X must have the same number of features as the training set."
            )

        
        n_samples = X.shape[0]
        log_scores = np.zeros((n_samples, self.n_classes))

        for (class_idx, current_class) in enumerate(self.classes):
            log_scores[:, class_idx] = np.log(
                self.prior_prob[class_idx]
            )

            for feature_idx in range(self.n_features):
                category_probs = self.feature_probs[
                    current_class
                ][feature_idx]

                probabilities = np.array([
                    category_probs[value]
                    for value in X[:, feature_idx]
                ])

                log_scores[:, class_idx] += np.log(probabilities)

        return log_scores
        
    def predict(self, X):
        log_scores = self._joint_log_proba(X)
        class_indexes = np.argmax(log_scores, axis=1)

        return self.classes[class_indexes]
        
        
        
    def predict_proba(self, X):
        log_scores = self._joint_log_proba(X)

        row_max = np.max(log_scores, axis=1, keepdims=True)
        scores = np.exp(log_scores - row_max)

        return scores / scores.sum(axis=1, keepdims=True)
