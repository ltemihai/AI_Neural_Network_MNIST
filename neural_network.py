import numpy as np
from scipy.stats import truncnorm


def ELU(x, alpha=0.01):
    return 1/(1 + np.exp(-x))


def ELU_derivated(x, alpha=0.01):
    return x * (1 - x)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)


def error(prediction, target):
    sum_score = 0
    new_target = np.reshape(target, (1, target.shape[0]))
    for i in range(len(new_target)):
        for j in range(len(new_target[i])):
            sum_score += new_target[i][j] * np.log(1e-15 + prediction[i][j])
    mean_sum_score = 1.0 / len(target) * sum_score
    return -mean_sum_score


def cross_entropy(prediction, target):
    samples = target.shape[0]
    return (prediction - target)/samples


def get_accuracy(x, y, model):
    accuracy = 0
    for xx, yy in zip(x, y):
        s = model.generate_result(xx)
        if s == np.argmax(yy):
            accuracy += 1
    return accuracy/len(x) * 100


class NeuralNetwork:
    def __init__(self, in_nodes, out_nodes, hidden_nodes, learning_rate):
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate
        self.generate_weights()

    def generate_weights(self):
        self.in_h1_weights = np.random.randn(self.in_nodes.shape[1], self.hidden_nodes)
        self.in_h1_bias = np.zeros((1, self.hidden_nodes))
        self.h1_h2_weights = np.random.randn(self.hidden_nodes, self.hidden_nodes)
        self.h1_h2_bias = np.zeros((1, self.hidden_nodes))
        self.h2_out_weights = np.random.randn(self.hidden_nodes, self.out_nodes.shape[1])
        self.h2_out_bias = np.zeros((1, self.out_nodes.shape[1]))

    # COMPUTES f(X) = (X * w) + B and passes it through activation function, ELU this case
    def feedforward(self, x):
            z1 = np.dot(x, self.in_h1_weights) + self.in_h1_bias
            self.in_h1_activation = ELU(z1)
            z2 = np.dot(self.in_h1_activation, self.h1_h2_weights) + self.h1_h2_bias
            self.h1_h2_activation = ELU(z2)
            z3 = np.dot(self.h1_h2_activation, self.h2_out_weights) + self.h2_out_bias
            self.h2_out_activation = ELU(z3)


    def backpropagation(self,x, y, epoch):

        delta_h2_out = cross_entropy(self.h2_out_activation, y)
        delta_z2 = np.dot(delta_h2_out, self.h2_out_weights.T)
        delta_h1_h2 = delta_z2 * ELU_derivated(self.h1_h2_activation)
        delta_z1 = np.dot(delta_h1_h2, self.h1_h2_weights.T)
        delta_in_h1 = delta_z1 * ELU_derivated(self.in_h1_activation)

        self.h2_out_weights -= self.learning_rate * np.dot(self.h1_h2_activation.T, delta_h2_out)
        self.h2_out_bias -= self.learning_rate * np.sum(delta_h2_out, axis=0, keepdims=True)
        self.h1_h2_weights -= self.learning_rate * np.dot(self.in_h1_activation.T, delta_h1_h2)
        self.h1_h2_bias -= self.learning_rate * np.sum(delta_h1_h2, axis=0)
        if epoch == 0:
            self.in_h1_weights -= self.learning_rate * np.dot(np.reshape(x, (len(x), 1)), delta_in_h1)
        else:
            self.in_h1_weights -= self.learning_rate * np.dot(x, delta_in_h1)
        self.in_h1_bias -= self.learning_rate * np.sum(delta_in_h1, axis=0)


    def generate_result(self, data):
        self.in_nodes = data
        self.feedforward(data)
        return self.h2_out_activation.argmax()



