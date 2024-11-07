#import torch
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import torch
import math


def get_model(name):
    if name == "linear":
        return Linear()
    else:
        raise ValueError(f"Model kind {name} not implemented")
    

class SKLearnModel:
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

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


def setup_training(model, optimizer="adam", lr=0.0001, lr_scheduler=None):
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else: # sgd
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if lr_scheduler is not None:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    else:
        lr_scheduler = None
    loss = torch.nn.BCELoss()
    return optimizer, loss, lr_scheduler


def train_model(model, X_train, y_train, n_epochs=100, optimizer="adam", lr=0.0001, lr_scheduler=None):
    optimizer, loss, lr_scheduler = setup_training(model, optimizer=optimizer, lr=lr, lr_scheduler=lr_scheduler)
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss_val = loss(y_pred, y_train)
        loss_val.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        #if epoch % 10 == 0:
        #    print(f"Epoch {epoch}: Loss {loss.item()}")
    model.eval()


class SimpleTorchModel(torch.nn.Module):
    def __init__(self, model):
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.cuda()


    def fit(self, X_train, y_train):
        X_train = torch.tensor(X_train).to(self.model.device)
        y_train = torch.tensor(y_train).to(self.model.device)
        # shuffle X_train and y_train
        indices = torch.randperm(X_train.shape[0])
        X_train = X_train[indices]
        y_train = y_train[indices]
        train_model(self.model, X_train, y_train, optimizer=self.optimizer, lr=self.lr, lr_scheduler=self.lr_scheduler)

    def predict_proba(self, X):
        X = torch.tensor(X).to(self.model.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X)
        return y_pred.cpu().numpy()
    
    def predict(self, X):
        return self.predict_proba(X) > 0.5
    
    def __forward__(self, X):
        return self.model(X)
    
    def save(self, path, name="model"):
        torch.save(self.model.state_dict(), path+f"/{name}.pth")

    def load(self, path, name="model"):
        self.model.load_state_dict(torch.load(path+f"/{name}.pth"))
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def __str__(self):
        return self.name


class SimpleRNN(SimpleTorchModel):
    def __init__(self, input_size, hidden_size, n_layers=3, dropout=0.0):
        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.rnn = torch.nn.RNN(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.output = torch.nn.Linear(hidden_size, 1)
        model = torch.nn.Sequential(self.linear, self.rnn, self.output)
        super().__init__(model)
        self.name = f"rnn-hs-{hidden_size}-nl-{n_layers}-d-{dropout}"


class SimpleGRU(SimpleTorchModel):
    def __init__(self, input_size, hidden_size, n_layers=3, dropout=0.0):
        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.output = torch.nn.Linear(hidden_size, 1)
        model = torch.nn.Sequential(self.linear, self.gru, self.output)
        super().__init__(model)
        self.name = f"gru-hs-{hidden_size}-nl-{n_layers}-d-{dropout}"


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class Transformer(SimpleTorchModel):
    def __init__(self, input_size, hidden_size, n_layers=3, dropout=0.0, max_length=50):
        self.pos = PositionalEncoding(hidden_size, max_length)
        self.decoder = torch.nn.TransformerDecoderLayer(d_model=hidden_size, nhead=1, dim_feedforward=hidden_size, dropout=dropout)
        self.mlp = torch.nn.Linear(input_size, hidden_size)

    def __forward__(self, X):
        # X is of shape (Seq Length, 4096). Must add positional encoding along axis 0 i.e. sequence length 
        raise ValueError("Not implemented") # This model probably should not be a SimpleTorchModel



def get_model_suite(suite_name):
    assert suite_name in ["base", "linear", "tree", "mlp", "transformer"]
    models = {}
    if suite_name == "linear":
        models["linear-pure"] = Linear(penalty="l2")
        for penalty in ["l1", "l2", "elasticnet"]:
            for C in [0.25, 0.5, 1.0, 2.0]:
                model = Linear(penalty=penalty, C=C, solver="saga")
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
        models["linear-pure"] = Linear(penalty="l2")
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