import numpy as np
from matplotlib import pyplot as plt
from mnist import load_mnist
from mlp import MLP

x_train, y_train, x_test, y_test = load_mnist()


x_train = x_train.reshape(-1, 784).astype(np.float32) / 255
x_test = x_test.reshape(-1,784).astype(np.float32) / 255
y_train_onehot = np.eye(10)[y_train]
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, y_train_onehot.shape)
mlp = MLP(input_size=784, hidden_size=128, output_size=10)

mlp.fit(x_train, y_train_onehot)

y_pred = mlp.predict(x_test)

accuracy = np.mean(y_pred == y_test)

print(f"Accuracy: {accuracy:.2f}")

