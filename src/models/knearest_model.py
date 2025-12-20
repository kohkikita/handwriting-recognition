from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import joblib   

class KNearestModel:
    def __init__(self, n_neighbors=5,weights='distance', metric ='euclidean'):
       self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric, n_jobs=-1)
       self.is_trained = False

    def fit(self, X,y):
        self.model.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        return self.model.predict(X)
    
    def score(self, X, y):
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring.")
        return self.model.score(X, y)
    
    def save(self, filepath):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")
        joblib.dump(self.model, filepath)

    @classmethod
    def load(cls, filepath):
        model = joblib.load(filepath)
        instance = cls()
        instance.model = model
        instance.is_trained = True
        return instance