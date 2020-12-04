import matplotlib.pyplot as plt
import numpy as np


class MaximumEntropyClassifier:
    def __init__(self, num_iter):
        self.W = None
        self.b = None
        self.num_iter = num_iter

    def feed_forward(self, X, w, b):
        f = np.dot(X, w) + b
        f = np.exp(f)
        denominator = f.sum(axis=1)
        for i in range(len(denominator)):
            f[i, :] /= denominator[i]
        return f

    def cost_function(self, p, Y, w, regularizer=0.8):
        return -1 * (np.sum(Y * np.log(p)) / len(Y)) + ((regularizer / 2) * np.sum(np.square(w)))

    def gradient(self, X, Y, p, regularizer=0.8):
        dw = (np.dot(X.T, (p - Y)) / len(X)) + regularizer * self.W
        db = sum(p - Y) / len(X)
        return dw, db

    def gradient_checking(self, X, Y, epsilon=10e-4):
        w = self.W.flatten()
        b = self.b.flatten()
        t = np.concatenate((w, b))
        m = len(b)
        n = len(w)
        delta_t = []
        for i in range(len(t)):
            backup = t[i]
            t_plus = t[i] + epsilon
            t[i] = t_plus
            p = self.feed_forward(X, np.array(t[:-m]).reshape(self.W.shape),
                                  np.array(t[n:]).reshape(self.b.shape))
            j_plus = self.cost_function(p, Y, np.array(t[:-m]).reshape(self.W.shape))
            t[i] = backup
            t_minus = t[i] - epsilon
            t[i] = t_minus
            p = self.feed_forward(X, np.array(t[:-m]).reshape(self.W.shape),
                                  np.array(t[n:]).reshape(self.b.shape))
            j_minus = self.cost_function(p, Y, np.array(t[:-m]).reshape(self.W.shape))
            t[i] = backup
            dj = (j_plus - j_minus) / (2 * epsilon)
            delta_t.append(dj)
        return np.array(delta_t[:-m]).reshape(self.W.shape), np.array(delta_t[n:]).reshape(
            self.b.shape)

    def train(self, X, Y, learning_rate=0.01):
        self.checking(X, Y)
        self.error_rate = []
        for i in range(self.num_iter):
            p = self.feed_forward(X, self.W, self.b)
            j = self.cost_function(p, Y, self.W)
            dw, db = self.gradient(X, Y, p)
            self.W -= learning_rate * dw
            self.b -= learning_rate * db
            self.error_rate.append(j)
        print("Accuracy", self.accuracy(p, Y))

    def accuracy(self, p, y):
        accuracy = 0
        for i in range(len(p)):
            if np.argmax(p[i]) == np.argmax(y[i]):
                accuracy += 1
        return 100*accuracy / len(p)

    def checking(self, X, Y):
        N, D = X.shape
        k = len(np.unique(Y))
        self.W = np.random.randn(D, k)
        self.b = np.random.randn(k)
        print(self.W, self.b)
        p = self.feed_forward(X, self.W, self.b)
        dw, db = self.gradient(X, Y, p)
        delta_w, delta_b = self.gradient_checking(X, Y)
        diff_w = abs(dw - delta_w)
        for i in diff_w:
            for j in i:
                if j < 1e-4:
                    print("Correct")
        for i in range(len(db)):
            diff = abs(db[i] - delta_b[i])
            if diff < 1e-4:
                print("Correct")


def main():
    with open('xor.dat', 'r') as f:
        data = [i.strip().split(",") for i in f.readlines()]
    data = np.array(data).astype(float)
    X = data[:, : -1]
    Y = data[:, -1]
    X = np.array(X)
    Y = np.eye(2)[np.array(Y).astype(int).reshape(-1)]
    print(Y)
    model = MaximumEntropyClassifier(500)
    model.train(X, Y)
    plt.plot(model.error_rate)
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.show()


if __name__ == '__main__':
    main()
