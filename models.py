#import torch
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import torch


def get_model(name):
    if name == "linear":
        return Linear()
    else:
        raise ValueError(f"Model kind {name} not implemented")
    

class SKLearnModel:
    def fit(self, X_train, y_train):
        raise NotImplementedError

    def score(self, X_train, y_train):
        return self.model.score(X_train, y_train)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X):
        return self.model.predict(X) > 0.5
    
    def save(self, path, name="model"):
        with open(path+f"/{name}.pkl", "wb") as f:
            pickle.dump(self.model, f)
    
    def load(self, path, name="model"):
        with open(path+f"/{name}.pkl", "rb") as f:
            self.model = pickle.load(f) 

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name


class Linear(SKLearnModel):
    def __init__(self, penalty='l2', C=1.0):
        self.name = f"linear-{penalty}-C-{C}"
        self.model = LogisticRegression(random_state=0, penalty=penalty, C=C)


class RandomForest(SKLearnModel):
    def __init__(self, n_estimators=100, max_depth=None):
        self.name = f"rf-n-{n_estimators}-d-{max_depth}"
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)


class MLP(SKLearnModel):
    def __init__(self, hidden_layer_sizes=(5000,), activation='relu', solver='adam', alpha=0.0001, lr=0.001):
        self.name = f"mlp-hls-{hidden_layer_sizes}-a-{activation}-s-{solver}-lr-{lr}"
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, learning_rate_init=lr)



class TorchModel:
    def __init__(self, model, n_epochs=100, lr=0.001, lr_scheduler=None, optimizer="adam", batch_size=32):
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.n_epochs = n_epochs
        self.lr = lr
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else: # sgd
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)    
        if lr_scheduler is not None:
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.1)
        else:
            self.lr_scheduler = None
        self.loss = torch.nn.BCELoss()


    def fit(self, X_train, y_train):
        X_train = torch.tensor(X_train).to(self.model.device)
        y_train = torch.tensor(y_train).to(self.model.device)
        # shuffle 
        indices = torch.randperm(X_train.shape[0])
        X_train = X_train[indices]
        y_train = y_train[indices]
        self.model.train()
        for epoch in range(self.n_epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(X_train)
            loss_val = self.loss(y_pred, y_train)
            loss_val.backward()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            #if epoch % 10 == 0:
            #    print(f"Epoch {epoch}: Loss {loss.item()}")
        self.model.eval()

    def predict_proba(self, X):
        X = torch.tensor(X).to(self.model.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X)
        return y_pred.cpu().numpy()
    
    def predict(self, X):
        return self.predict_proba(X) > 0.5
    
    def save(self, path, name="model"):
        torch.save(self.model.state_dict(), path+f"/{name}.pth")

    def load(self, path, name="model"):
        self.model.load_state_dict(torch.load(path+f"/{name}.pth"))
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def __str__(self):
        return self.name


class RNN(TorchModel):
    pass


class GRU(TorchModel):
    pass


class Transformer(TorchModel):
    pass



def get_model_suite(suite_name):
    assert suite_name in ["base", "linear", "tree", "mlp", "transformer"]
    models = {}
    if suite_name == "linear":
        models["linear-pure"] = Linear(penalty="none")
        for penalty in ["l1", "l2", "elasticnet"]:
            for C in [0.25, 0.5, 1.0, 2.0]:
                model = Linear(penalty=penalty, C=C)
                models[model.name] = model

    elif suite_name == "tree":
        for n_estimators in [10, 50, 100, 200]:
            for max_depth in [None, 10, 20, 50]:
                model = RandomForest(n_estimators=n_estimators, max_depth=max_depth)
                models[model.name] = model

    elif suite_name == "mlp":
        for hidden_layer_sizes in [(100,), (500,), (1000,), (5000,)]:
            for activation in ["relu", "tanh"]:
                for solver in ["adam", "sgd"]:
                    for lr in [0.001, 0.01]:
                        model = MLP(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, lr=lr)
                        models[model.name] = model

    else:
        models["linear-pure"] = Linear(penalty="none")
        for penalty in ["l2"]:
            for C in [0.5, 1.0]:
                model = Linear(penalty=penalty, C=C)
                models[model.name] = model
        for n_estimators in [50]:
            for max_depth in [10, 20]:
                model = RandomForest(n_estimators=n_estimators, max_depth=max_depth)
                models[model.name] = model
        for hidden_layer_sizes in [(5000, 10_000)]:
            for activation in ["relu"]:
                for solver in ["adam", "sgd"]:
                    for lr in [0.001, 0.01]:
                        model = MLP(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, lr=lr)
                        models[model.name] = model
    return models