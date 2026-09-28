import numpy as np
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2 / self.input_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2 / self.hidden_size)

        self.bias1 = np.zeros((1, self.hidden_size))
        self.bias2 = np.zeros((1, self.output_size))

    def fit(self, X, y, epochs = 10, batch_size = 64):
        for epoch in range(epochs):
            order = np.random.permutation(len(X))
            for start in range(0, len(X), batch_size):
                idx = order[start:start + batch_size]
                Xb, yb = X[idx], y[idx]

                # feedforward
                layer1 = Xb.dot(self.weights1) + self.bias1
                activation1 = ReLU(layer1)
                layer2 = activation1.dot(self.weights2) + self.bias2
                activation2 = softmax(layer2)

                #backpropagation
                error = (activation2 - yb) / len(Xb)
                d_weights2 = activation1.T.dot(error)
                d_bias2 = np.sum(error, axis = 0, keepdims=True)
                error_hidden = error.dot(self.weights2.T) * dReLU(layer1)
                d_weights1 = Xb.T.dot(error_hidden)
                d_bias1 = np.sum(error_hidden, axis = 0, keepdims=True)

                self.weights2 -= self.learning_rate * d_weights2
                self.weights1 -= self.learning_rate * d_weights1
                self.bias2 -= self.learning_rate * d_bias2
                self.bias1 -= self.learning_rate * d_bias1
            print(f"epoch {epoch+1}/{epochs} done")

    def predict(self, X):
        layer1 = X.dot(self.weights1) + self.bias1
        activation1 = ReLU(layer1)
        layer2 = activation1.dot(self.weights2) + self.bias2
        return np.argmax(layer2, axis=1)
        
def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)