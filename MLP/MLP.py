import numpy as np
import matplotlib.pyplot as plt

def sigmoid_forward(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(x):
    x = sigmoid_forward(x)
    return x * (1 - x)


class MLP:
    def __init__(self, input_dim, output_dim, hidden_dim=16, activation='sigmoid', lr=0.01):
        self.init_layers(input_dim, output_dim, hidden_dim)
        self.init_activation(activation)
        self.lr = lr

    def init_layers(self, input_dim, output_dim, hidden_dim):
        self.params = {}
        
        self.params['W1'] = np.random.normal(0, np.sqrt(1/input_dim), size=(hidden_dim, input_dim))
        self.params['b1'] = np.random.normal(0, np.sqrt(1/hidden_dim), size=(hidden_dim, 1))

        self.params['W2'] = np.random.normal(0, np.sqrt(1/input_dim), size=(output_dim, hidden_dim))
        self.params['b2'] = np.random.normal(0, np.sqrt(1/input_dim), size=(output_dim, 1))

    def init_activation(self, activation):
        if activation == 'sigmoid':
            self.activation_forward = sigmoid_forward
            self.activation_backward = sigmoid_backward

    def forward(self, X):
        self.memory = {}
        self.memory['X'] = x
        
        self.memory['Z'] = np.matmul(self.params.get('W1'), X.T) + self.params.get('b1')
        self.memory['H'] = self.activation_forward(self.memory.get('Z'))

        self.memory['Y'] = np.matmul(self.params.get('W2'), self.memory.get('H')) + self.params.get('b2')
        return self.memory.get('Y').T

    def backward(self, y):
        self.grad_values = {}

        dJ_dY = self.memory.get('Y').T - y.reshape(-1, 1)

        dJ_dW2 = np.zeros_like(self.params.get('W2'))
        dJ_db2 = np.zeros_like(self.params.get('b2'))
        h = self.memory.get('H').T

        for i in range(y.shape[0]):
            dJ_dW2 += np.outer(dJ_dY[i], h[i])
            dJ_db2 += dJ_dY[i].reshape(-1, 1)

        dJ_dW2 = dJ_dW2 / y.shape[0]
        dJ_db2 = dJ_db2 / y.shape[0]

        dJ_dW1 = np.zeros_like(self.params.get('W1'))
        dJ_db1 = np.zeros_like(self.params.get('b1'))
        x = self.memory.get('X')
        z = self.memory.get('Z').T

        for i in range(y.shape[0]):
            dJ_dW1 += np.outer(np.matmul(self.params.get('W2').T, dJ_dY[i].reshape(-1, 1)) * self.activation_backward(z[i].reshape(-1, 1)), x[i])
            dJ_db1 += np.matmul(self.params.get('W2').T, dJ_dY[i].reshape(-1, 1)) * self.activation_backward(z[i].reshape(-1, 1))

        dJ_dW1 = dJ_dW1 / y.shape[0]
        dJ_db1 = dJ_db1 / y.shape[0]

        self.grad_values['W1'] = dJ_dW1
        self.grad_values['b1'] = dJ_db1
        self.grad_values['W2'] = dJ_dW2
        self.grad_values['b2'] = dJ_db2

        self.update_params()

    def update_params(self):

        self.params['W1'] = self.params.get('W1') - self.lr * self.grad_values.get('W1')
        self.params['b1'] = self.params.get('b1') - self.lr * self.grad_values.get('b1')

        self.params['W2'] = self.params.get('W2') - self.lr * self.grad_values.get('W2')
        self.params['b2'] = self.params.get('b2') - self.lr * self.grad_values.get('b2')


if __name__ == '__main__':
    x = np.arange(0, 10, 1).reshape(-1, 1)
    y = np.arange(0, 10, 1) * 0.5

    model = MLP(input_dim=1, output_dim=1, hidden_dim=16)

    num_iters = 10000
    for i in range(num_iters):
        pred_y = model.forward(x)
        model.backward(y)

    y_hat = model.forward(x)
    plt.style.use('ggplot')
    plt.figure(figsize=(10.8, 7.2))
    plt.scatter(x, y, s=10, color='crimson', label='true_y')
    plt.scatter(x, y_hat, s=10, color='navy', label='pred_y')
    plt.legend(fontsize='large')
    plt.show()