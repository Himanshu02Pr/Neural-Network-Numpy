import numpy as np

def one_hot(Y, num_classes):
    return np.eye(num_classes)[Y]

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    exps = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def ReLU_deriv(Z):
    return (Z > 0).astype(float)


class Simple_NN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward_prop(self, X):
        self.Z1 = X@self.W1 + self.b1
        self.A1 = ReLU(self.Z1)
        self.Z2 = self.A1@self.W2 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2
    
    def compute_loss(Y_hat, Y):
        m = Y.shape[0]
        return -np.sum(Y * np.log(Y_hat + 1e-8)) / m

    def backward_prop(self, X, Y):
        m = X.shape[0]

        dZ2 = self.A2 - Y
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * ReLU_deriv(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, X):
        probabs = self.forward_prop(X)
        return np.argmax(probabs, axis=1)