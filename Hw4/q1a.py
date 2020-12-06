import matplotlib.pyplot as plt
import numpy as np


class MLP:
    def __init__(self, num_iter):
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.num_iter = num_iter

    def sigmoid(self, z):
        return 1 / 1 + np.exp(-z)

    def sigmoid_der(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self, A):
        expA = np.exp(A - np.max(A))
        return expA / expA.sum(axis=1, keepdims=True)

    def cost_function(self, p, Y, regularizer = 0.8):
        return -1 * (np.sum(Y * np.log(p)) / len(Y)) + ((regularizer / 2) * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2))))/len(Y)

    def gradient_checking2(self, X, Y, epsilon=1e-4):
        w1 = self.W1.flatten()
        b1 = self.b1.flatten()
        t1 = np.concatenate((w1, b1))
        w2 = self.W2.flatten()
        b2 = self.b2.flatten()
        t2 = np.concatenate((w2, b2))
        m1 = len(b1)
        n1 = len(w1)
        m2 = len(b2)
        n2 = len(w2)
        delta_t1 = []
        delta_t2 = []
        for i in range(len(t1)):
            backup = t1[i]
            t_plus = t1[i] + epsilon
            t1[i] = t_plus
            Z1 = np.dot(X, np.array(t1[:-m1]).reshape(self.W1.shape)) + np.array(t1[n1:]).reshape(self.b1.shape)
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, np.array(t2[:-m2]).reshape(self.W2.shape)) + np.array(t2[n2:]).reshape(self.b2.shape)
            A2 = self.softmax(Z2)
            j_plus = self.cost_function(A2, Y)
            t1[i] = backup
            t_minus = t1[i] - epsilon
            t1[i] = t_minus
            Z1 = np.dot(X, np.array(t1[:-m1]).reshape(self.W1.shape)) + np.array(t1[n1:]).reshape(self.b1.shape)
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, np.array(t2[:-m2]).reshape(self.W2.shape)) + np.array(t2[n2:]).reshape(self.b2.shape)
            A2 = self.softmax(Z2)
            j_minus = self.cost_function(A2, Y)
            t1[i] = backup
            dj = (j_plus - j_minus) / (2 * epsilon)
            delta_t1.append(dj)
        for i in range(len(t2)):
            backup = t2[i]
            t_plus = t2[i] + epsilon
            t2[i] = t_plus
            Z1 = np.dot(X, np.array(t1[:-m1]).reshape(self.W1.shape)) + np.array(t1[n1:]).reshape(self.b1.shape)
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, np.array(t2[:-m2]).reshape(self.W2.shape)) + np.array(t2[n2:]).reshape(self.b2.shape)
            A2 = self.softmax(Z2)
            j_plus = self.cost_function(A2, Y)
            t2[i] = backup
            t_minus = t2[i] - epsilon
            t2[i] = t_minus
            Z1 = np.dot(X, np.array(t1[:-m1]).reshape(self.W1.shape)) + np.array(t1[n1:]).reshape(self.b1.shape)
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, np.array(t2[:-m2]).reshape(self.W2.shape)) + np.array(t2[n2:]).reshape(self.b2.shape)
            A2 = self.softmax(Z2)
            j_minus = self.cost_function(A2, Y)
            t2[i] = backup
            dj = (j_plus - j_minus) / (2 * epsilon)
            delta_t2.append(dj)
        return np.array(delta_t1[:-m1]).reshape(self.W1.shape), np.array(delta_t1[n1:]).reshape(self.b1.shape), \
               np.array(delta_t2[:-m2]).reshape(self.W2.shape), np.array(delta_t2[n2:]).reshape(self.b2.shape)

    def train(self, X, Y, learning_rate=0.02, regularizer=0.8):
        self.checking(X, Y)
        self.error_rate = []
        for i in range(self.num_iter):
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.softmax(Z2)
            dZ2 = A2 - Y
            dZ2_dW2 = A1
            dW2 = np.dot(dZ2_dW2.T, dZ2)/len(X) + np.multiply(regularizer, self.W2)/len(X)
            db2 = dZ2/len(X)
            dW1 = (np.dot(X.T, self.sigmoid_der(Z1) * np.dot(dZ2, self.W2.T)))/len(X) + np.multiply(regularizer, self.W1)/len(X)
            db1 = (np.dot(dZ2, self.W2.T) * self.sigmoid_der(Z1))/len(X)
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1.sum(axis=0)
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2.sum(axis=0)
            j = self.cost_function(A2, Y)
            self.error_rate.append(j)
        print("Accuracy", self.accuracy(A2, Y))

    def accuracy(self, p, y):
        accuracy = 0
        for i in range(len(p)):
            if np.argmax(p[i]) == np.argmax(y[i]):
                accuracy += 1
        return 100 * accuracy / len(p)

    def checking(self, X, Y):
        N, D = X.shape
        k = len(np.unique(Y))
        self.W1 = np.random.randn(D, 3) * 0.01
        self.b1 = np.zeros(3)
        self.W2 = np.random.randn(3, k) * 0.01
        self.b2 = np.zeros(k)
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)
        dZ2 = A2 - Y
        dZ2_dW2 = A1
        dW2 = np.dot(dZ2_dW2.T, dZ2)/len(X)
        db2 = dZ2/len(X)
        dW1 = (np.dot(X.T, self.sigmoid_der(Z1) * np.dot(dZ2, self.W2.T)))/len(X)
        db1 = (np.dot(dZ2, self.W2.T) * self.sigmoid_der(Z1))/len(X)
        delta_w1, delta_b1, delta_w2, delta_b2 = self.gradient_checking2(X, Y)
        b1_check = abs(db1 - delta_b1.sum(axis=0))
        b2_check = abs(db2 - delta_b2.sum(axis=0))
        w1_check = abs(dW1 - delta_w1)
        w2_check = abs(dW2 - delta_w2)
        for i in b1_check:
            for j in i:
                if j < 1e-3:
                    print("Correct")
                else:
                    print("Gradient check did not pass")
        for i in b2_check:
            for j in i:
                if j < 1e-3:
                    print("Correct")
                else:
                    print("Gradient check did not pass")
        for i in w1_check:
            for j in i:
                if j < 1e-3:
                    print("Correct")
                else:
                    print("Gradient check did not pass")
        for i in w2_check:
            for j in i:
                if j < 1e-3:
                    print("Correct")
                else:
                    print("Gradient check did not pass")


def main():
    with open('xor.dat', 'r') as f:
        data = [i.strip().split(",") for i in f.readlines()]
    data = np.array(data).astype(float)
    X = data[:, : -1]
    Y = data[:, -1]
    X = np.array(X)
    Y = np.eye(2)[np.array(Y).astype(int).reshape(-1)]
    model = MLP(100)
    # model.checking(X, Y)
    model.train(X, Y)
    plt.plot(model.error_rate)
    print(model.error_rate[-1])
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.show()


if __name__ == '__main__':
    main()
