#import torch
import pickle
from sklearn.linear_model import LogisticRegression

def get_model(name):
    if name == "linear":
        return Linear()
    else:
        raise ValueError(f"Model kind {name} not implemented")

class Linear:
    def __init__(self):
        self.model = LogisticRegression(random_state=0, penalty='l2')

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def score(self, X_train, y_train):
        return self.model.score(X_train, y_train)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, path):
        with open(path+"/model.pkl", "wb") as f:
            pickle.dump(self.model, f)
    
    def load(self, path):
        with open(path+"/model.pkl", "rb") as f:
            self.model = pickle.load(f)