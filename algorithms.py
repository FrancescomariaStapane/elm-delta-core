from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from ELM_CrossEntropy import *
from codecarbon import OfflineEmissionsTracker
import pandas as pd
import torch
import hpelm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import resource

INFINITY = 10**12
# INFINITY = 10**10
AS_LIMIT = 0.3 * 1024 ** 3

class MlAlgorithm(ABC):
    def __init__(self, xtr, ytr, xts, yts):
        self.xtr = xtr
        self.ytr = ytr
        self.xts = xts
        self.yts = yts
        self.valid = True
    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def refresh(self):
        pass
    @abstractmethod
    def get_default_accuracy(self):
        pass


class RandomForest(MlAlgorithm):
    def __init__(self, xtr, ytr, xts, yts, n_estimators, max_depth):
        super().__init__(xtr, ytr, xts, yts)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.refresh()

    def learn(self,):
        tracker = new_tracker()
        tracker.start()
        time_start = time.time_ns()
        self.random_forest_model.fit(self.xtr, self.ytr)
        tracker.stop()
        train_time = float((time.time_ns() - time_start) / 10 ** 6)  # in ms
        training_energy = tracker.final_emissions_data.energy_consumed
        return train_time, training_energy

    def refresh(self):
        self.random_forest_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=8,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=1,
            n_jobs=-1
        )

    def test(self):
        time, energy, y_pred = run_with_measurement(lambda: self.random_forest_model.predict(self.xts))
        accuracy = accuracy_score(self.yts, y_pred)
        f1 = f1_score(self.yts, y_pred, average='weighted')
        return time, energy, (accuracy, f1)
        # print(f'Accuratezza del modello: {accuracy:.4f}')
        # print(f'F1: {f1:.4f}')

    def get_default_accuracy(self):
        y = pd.DataFrame(self.yts)
        counts = y.value_counts(normalize=True)
        def_acc = counts.values.max()
        return def_acc
def numpy_to_float_tensor(items):
    items = torch.from_numpy(items)
    items = items.type('torch.FloatTensor')
    return items

def new_tracker():
    tracker = OfflineEmissionsTracker(
        country_iso_code="ITA",
        log_level="error",  # Suppresses detailed logging
        save_to_file=False  # Prevents writing to emissions.csv
    )
    return tracker

class CrossEntropyElm(MlAlgorithm):
    def __init__(self, xtr, ytr, xts, yts, n_neurons, learning_rate):
        # ytr = torch.from_numpy(ytr.argmax(axis = 1))
        # yts = torch.from_numpy(yts.argmax(axis = 1))
        super().__init__(xtr, ytr, xts, yts)
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.W, self.b = random_weights(len(self.xtr[0]), self.n_neurons)
        self.beta = []

    def refresh(self):
        self.W, self.b = random_weights(len(self.xtr[0]), self.n_neurons)

    def learn(self):
        # tracker = new_tracker()
        # tracker.start()
        # time_start = time.time_ns()
        # self.beta = training(self.n_neurons, self.xtr, self.ytr, self.W, self.b, self.learning_rate)
        # tracker.stop()
        # train_time = float((time.time_ns() - time_start) / 10 ** 6)  # in ms
        # training_energy = tracker.final_emissions_data.energy_consumed
        # return train_time, training_energy
        train_time, train_energy, self.beta = run_with_measurement(lambda: training(self.n_neurons, self.xtr, self.ytr, self.W, self.b, self.learning_rate))
        return train_time, train_energy

    def test(self):
        # accuracy, f1 = test_model(self.xtr, self.ytr, self.W, self.b, self.beta, silent=False)
        return run_with_measurement(lambda: test_model(self.xtr, self.ytr, self.W, self.b, self.beta, silent=False))
    def get_default_accuracy(self):
        most_common_class = torch.mode(self.yts).values
        default_accuracy = (self.yts == most_common_class).sum().item() / self.yts.size(0)
        return default_accuracy
def rae(y_real, y_pred):
    abs_error = 0
    avg_real = 0
    for i in range(y_real.shape[0]):
        abs_error += np.abs(y_real[i] - y_pred[i])
        avg_real += y_real[i]
    mae = abs_error / y_real.shape[0]
    avg_real = avg_real / y_real.shape[0]
    mad = 0
    for i in range(y_real.shape[0]):
        mad += np.abs(y_real[i] - avg_real)
    mad = mad / y_real.shape[0]
    return mae / mad

def mape (y_real, y_pred):
    mape = 0
    for i in range (y_real.shape[0]):
        delta = 0
        if (y_real[i]) == 0:
            delta = 0.00000001
        mape+=(np.abs(y_real[i] - y_pred[i]))/(y_real[i] + delta)
    return mape / y_real.shape[0] * 100
def sse(y_real, y_predict):
    sse = 0
    # try:
    #     return (y_real - y_predict) **2
    # except RuntimeError as e :
    #     print(e)
    for i in range(y_real.shape[0]):
        sse += (y_real[i] - y_predict[i])**2
    return sse
    y_real, y_predict = torch.tensor(y_real), torch.tensor(y_predict)
    # y_real, y_predict = y_real.detach.clone(), y_predict.detach.clone()
    return torch.sum((y_predict - y_real) ** 2)
def sigmoid(X, W, b):
    z = safe_matmul(X, W) + b
    return torch.sigmoid(z)

class Elm(MlAlgorithm):
    def __init__(self, xtr, ytr, xts, yts, n_neurons, lmbda):
        super().__init__(xtr, ytr, xts, yts)
        try:
            # print("limit: ", resource.getrlimit(resource.RLIMIT_AS))
            # target normalization
            self.interval = torch.cat([ytr, yts]).max() - torch.cat([ytr, yts]).min()
            self.target_normalization = True
            if self.target_normalization:
                self.y_mean = self.ytr.mean()
                self.y_std = self.ytr.std()
                self.ytr = (self.ytr - self.y_mean) / self.y_std
            self.input_dimension = len(self.xtr[0])
            self.n_neurons = n_neurons
            self.W = torch.randn(self.input_dimension, self.n_neurons)
            self.b = torch.randn(1, self.n_neurons)
            self.h = sigmoid(self.xtr, self.W, self.b)
            self.lmbda = lmbda
        except MemoryError:
            self.valid = False

    def learn(self):
        if self.valid:
            try:
                h_transpose = self.h.T
                train_time, train_energy, self.beta = run_with_measurement(lambda:
                    safe_matmul(
                    torch.inverse(safe_matmul(h_transpose, self.h) + self.lmbda * torch.eye(self.n_neurons)),
                    safe_matmul(h_transpose, self.ytr)
                ))
                A = safe_matmul(h_transpose, self.h) + self.lmbda * torch.eye(self.n_neurons)
                b = safe_matmul(h_transpose, self.ytr)
                self.beta = torch.linalg.solve(A, b)
                return train_time, train_energy
            except MemoryError:
                self.valid = False
        return INFINITY, INFINITY

    def test(self):
        if self.valid:
            h = sigmoid(self.xts, self.W, self.b)
            test_time, test_energy, y_pred = run_with_measurement(lambda: safe_matmul(h,self.beta))
            if self.target_normalization:
                y_pred = y_pred * self.y_std + self.y_mean
            loss = sse(self.yts , y_pred)
            mean_loss = loss / self.yts.size(0)
            rae_ = rae(self.yts, y_pred)
            return test_time, test_energy, (mean_loss, rae_)
        return INFINITY, INFINITY, (INFINITY, INFINITY)
    def refresh(self):
            self.valid = True
            self.W = torch.randn(self.input_dimension, self.n_neurons)
            self.b = torch.randn(1, self.n_neurons)
            try:
                self.h = sigmoid(self.xtr, self.W, self.b)
            except MemoryError:
                self.valid = False

    def get_default_accuracy(self):
        pass
class NN(MlAlgorithm):
    def __init__(self, xtr, ytr, xts, yts, n_neurons=[64, 32], lmbda=0.0,
                 lr=0.001, epochs=100, batch_size=32, activation=nn.ReLU, device="cpu"):
        super().__init__(xtr, ytr, xts, yts)
        if ytr.ndim == 1:
            ytr = ytr.unsqueeze(1)
        if yts.ndim == 1:
            yts = yts.unsqueeze(1)

        self.ytr = ytr
        self.yts = yts
        # target normalization
        self.target_normalization = True
        if self.target_normalization:
            self.y_mean = self.ytr.mean()
            self.y_std = self.ytr.std()
            self.ytr = (self.ytr - self.y_mean) / self.y_std

        self.input_dimension = xtr.size(1)
        self.output_dimension = ytr.size(1) if ytr.ndim > 1 else 1

        # neural network architecture
        layers = []
        in_dim = self.input_dimension
        for hidden_dim in n_neurons:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dimension))
        self.model = nn.Sequential(*layers)

        self.device = device
        self.model = self.model.to(self.device)

        self.lmbda = lmbda
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.lmbda)
        self.criterion = nn.MSELoss()

        # DataLoader for mini-batch training
        self.train_dataset = TensorDataset(self.xtr, self.ytr)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def learn(self):
        def train_loop():
            self.model.train()
            for epoch in range(self.epochs):
                for X_batch, y_batch in self.train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    self.optimizer.zero_grad()
                    y_pred = self.model(X_batch)
                    loss = self.criterion(y_pred, y_batch)
                    loss.backward()
                    self.optimizer.step()
            return loss.item()  # return last batch loss

        train_time, train_energy, final_loss = run_with_measurement(train_loop)
        return train_time, train_energy

    def test(self):
        X = self.xts.to(self.device)
        y_true = self.yts.to(self.device)

        def inference():
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X)
            return y_pred

        test_time, test_energy, y_pred = run_with_measurement(inference)

        if self.target_normalization:
            y_pred = y_pred * self.y_std + self.y_mean

        loss = sse(y_true, y_pred)
        mean_loss = loss / y_true.size(0)
        rae_ = rae(y_true, y_pred)

        return test_time, test_energy, (mean_loss, rae_)

    def refresh(self):
        # reinitialize network weights
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        self.model.apply(weight_reset)

    def get_default_accuracy(self):
        # R² score
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.xts.to(self.device))
            if self.target_normalization:
                y_pred = y_pred * self.y_std + self.y_mean
            ss_res = torch.sum((self.yts - y_pred.cpu()) ** 2)
            ss_tot = torch.sum((self.yts - self.yts.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot
        return r2.item()

class LR(MlAlgorithm):
    def __init__(self, xtr, ytr, xts, yts, lmbda=0.0):
        super().__init__(xtr, ytr, xts, yts)
        self.interval = torch.cat([ytr, yts]).max() - torch.cat([ytr, yts]).min()

        # target normalization
        self.target_normalization = True
        if self.target_normalization:
            self.y_mean = self.ytr.mean()
            self.y_std = self.ytr.std()
            self.ytr = (self.ytr - self.y_mean) / self.y_std

        self.input_dimension = xtr.size(1)
        self.lmbda = lmbda


    def learn(self):
        X = self.xtr
        y = self.ytr

        # Normal Equation with optional ridge regularization
        XTX = safe_matmul(X.T, X) + self.lmbda * torch.eye(self.input_dimension)
        XTy = safe_matmul(X.T, y)

        train_time, train_energy, beta = run_with_measurement(
            lambda: torch.linalg.solve(XTX, XTy)
        )
        self.beta = beta
        return train_time, train_energy

    def test(self):
        X = self.xts
        test_time, test_energy, y_pred = run_with_measurement(lambda: safe_matmul(X, self.beta))

        if self.target_normalization:
            y_pred = y_pred * self.y_std + self.y_mean

        loss = sse(self.yts, y_pred)
        mean_loss = loss / self.yts.size(0)
        rae_ = rae(self.yts, y_pred)

        return test_time, test_energy, (mean_loss, rae_)

    def refresh(self):
        # No hidden weights to reinitialize in Linear Regression,
        # but we can just reset beta so retraining is required.
        self.beta = None

    def get_default_accuracy(self):
        # A reasonable default: return R² score
        with torch.no_grad():
            y_pred = safe_matmul(self.xts, self.beta)
            if self.target_normalization:
                y_pred = y_pred * self.y_std + self.y_mean

            ss_res = torch.sum((self.yts - y_pred) ** 2)
            ss_tot = torch.sum((self.yts - self.yts.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot
        return r2.item()
# class BatchElm(MlAlgorithm):
#     def __init__(self, xtr, ytr, xts, yts, n_neurons, lmbda):
#         super().__init__(xtr, ytr, xts, yts)
#         # target normalization
#         self.interval = torch.cat([ytr, yts]).max() - torch.cat([ytr, yts]).min()
#         self.target_normalization = True
#         if self.target_normalization:
#             self.y_mean = self.ytr.mean()
#             self.y_std = self.ytr.std()
#             self.ytr = (self.ytr - self.y_mean) / self.y_std
#         self.input_dimension = len(self.xtr[0])
#         self.n_neurons = n_neurons
#         self.W = torch.randn(self.input_dimension, self.n_neurons)
#         self.b = torch.randn(1, self.n_neurons)
#         self.h = sigmoid(self.xtr, self.W, self.b)
#         self.lmbda = lmbda
#         self.P = torch.inverse(self.lmbda * torch.eye(self.n_neurons))
#         self.beta = torch.zeros(self.n_neurons, 1)
#
#     def learn(self):
#         h_transpose = self.h.T
#         train_time, train_energy, self.beta = run_with_measurement(lambda:
#
#                                                                    )
#         A = torch.matmul(h_transpose, self.h) + self.lmbda * torch.eye(self.n_neurons)
#         b = torch.matmul(h_transpose, self.ytr)
#         self.beta = torch.linalg.solve(A, b)
#         return train_time, train_energy
#
#     def test(self):
#         h = self.compute_hidden(self.xts, batch_size=128)
#         test_time, test_energy, y_pred = run_with_measurement(lambda: torch.matmul(h, self.beta))
#         if self.target_normalization:
#             y_pred = y_pred * self.y_std + self.y_mean
#         loss = sse(self.yts, y_pred)
#         mean_loss = loss / self.yts.size(0)
#         mean_value = sum(self.yts) / self.yts.size(0)
#         return test_time, test_energy, mean_loss
#
#     def refresh(self):
#         self.W = torch.randn(self.input_dimension, self.n_neurons)
#         self.b = torch.randn(1, self.n_neurons)
#         self.h = sigmoid(self.xtr, self.W, self.b)
#
#
#     def compute_hidden(self, x, batch_size=128):
#         H_batches = []
#         for i in range(0, x.size(0), batch_size):
#             xb = x[i:i + batch_size]
#             Hb = torch.tanh(xb @ self.W + self.b)  # or sigmoid
#             H_batches.append(Hb)
#         return torch.cat(H_batches, dim=0)
#     def get_default_accuracy(self):
#         pass
#
#     def learn_batch(self, x_batch, y_batch):
#         Hb = torch.tanh(x_batch @ self.W + self.b)  # hidden activations
#         HbT = Hb.T
#
#         # Update P (matrix inverse lemma)
#         temp = torch.inverse(torch.eye(Hb.size(0)) + Hb @ self.P @ HbT)
#         self.P = self.P - self.P @ HbT @ temp @ Hb @ self.P
#
#         # Update beta
#         self.beta = self.beta + self.P @ HbT @ (y_batch - Hb @ self.beta)

class HpElmRegression(MlAlgorithm):
    def __init__(self, xtr, ytr, xts, yts, n_neurons, t_neurons):
        super().__init__(xtr, ytr, xts, yts)
        self.xtr = torch.Tensor.numpy(xtr)
        self.ytr = torch.Tensor.numpy(ytr)
        self.xts = torch.Tensor.numpy(xts)
        self.yts = torch.Tensor.numpy(yts)
        self.target_normalization = True
        self.interval = torch.cat([ytr, yts]).max() - torch.cat([ytr, yts]).min()
        if self.target_normalization:
            self.y_mean = self.ytr.mean()
            self.y_std = self.ytr.std()
            self.ytr = (self.ytr - self.y_mean) / self.y_std
        self.n_neurons = n_neurons
        self.t_neurons = t_neurons
        self.model = hpelm.HPELM(self.xtr.shape[1], 1)
        self.model.add_neurons(self.n_neurons, self.t_neurons)

    def learn(self):
        train_time, train_energy, results = run_with_measurement(lambda: self.model.train(self.xtr, self.ytr))
        return train_time, train_energy
    def test(self):

        test_time, test_energy,y_pred = run_with_measurement(lambda: self.model.predict(self.xts))
        if self.target_normalization:
            y_pred = y_pred * self.y_std + self.y_mean
        loss = sse(self.yts , y_pred)
        mean_loss = loss.mean()
        mean_value = self.yts.mean()
        return test_time, test_energy, mean_loss
    def refresh(self):
        self.model = hpelm.HPELM(self.xtr.shape[1], 1)
        self.model.add_neurons(self.n_neurons, self.t_neurons)
    def get_default_accuracy(self):
        pass


def run_with_measurement(code):
    tracker = new_tracker()
    tracker.start()
    time_start = time.time_ns()
    results = code()
    tracker.stop()
    time_passed = float((time.time_ns() - time_start) / 10 ** 6)  # in ms
    energy_used = tracker.final_emissions_data.energy_consumed
    return time_passed, energy_used, results

def estimate_matmul_memory(A: torch.Tensor, B: torch.Tensor, safety_factor=1.3):
    dtype_size = torch.tensor([], dtype=A.dtype).element_size()
    size_A = A.numel() * dtype_size
    size_B = B.numel() * dtype_size
    m, n, n2, p= 1,1,1,1
    if(A.dim() == 1):
        m = A.shape[0]
    else:
        m, n = A.shape[0], A.shape[1]
    if (B.dim() == 1):
        p = B.shape[0]
    else:
        n2, p = B.shape[0], B.shape[1]
    # assert n == n2, "incompatible shapes"
    size_out = m * p * dtype_size
    total = (size_A + size_B + size_out) * safety_factor
    return total

def safe_matmul(A: torch.Tensor, B:torch.Tensor, safety_factor=1.3, limit=AS_LIMIT):
    if(estimate_matmul_memory(A,B, safety_factor) > limit):
        print("aborting matmul, ", (estimate_matmul_memory(A,B, safety_factor)))
        raise MemoryError
    return torch.matmul(A, B)