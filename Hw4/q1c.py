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
        expA = np.exp(A)
        return expA / expA.sum(axis=1, keepdims=True)

    def cost_function(self, p, Y, regularizer=0.6):
        return -1 * (np.sum(Y * np.log(p)) / len(Y)) + (
            (regularizer/2) * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2))))/len(Y)

    def accuracy(self, p, y):
        accuracy = 0
        for i in range(len(p)):
            if np.argmax(p[i]) == np.argmax(y[i]):
                accuracy += 1
        return accuracy / len(p)

    def mini_batch_train(self, learning_rate=0.0006, regularizer=0.6):
        global avg
        with open('iris_train.dat', 'r') as f:
            data = [i.strip().split(",") for i in f.readlines()]
        data = np.array(data).astype(float)
        with open('iris_test.dat', 'r') as f:
            test_data = [i.strip().split(",") for i in f.readlines()]
        test_data = np.array(test_data).astype(float)
        X_test = test_data[:, :-1]
        Y_test = test_data[:, -1]
        Y_test = np.eye(3)[np.array(Y_test).astype(int).reshape(-1)]
        N, D = data[:, :-1].shape
        k = len(np.unique(data[:, -1]))
        self.W1 = np.random.randn(D, 20) * 0.01
        self.b1 = np.zeros(20)
        self.W2 = np.random.randn(20, k) * 0.01
        self.b2 = np.zeros(k)
        self.train_error = []
        self.test_error = []
        self.train_accuracy = []
        self.test_accuracy = []
        for i in range(self.num_iter):
            batch = np.split(data, 5)
            avg_l = 0
            avg_a = 0
            for b in batch:
                X = b[:, : -1]
                Y = b[:, -1]
                X = np.array(X)
                Y = np.eye(3)[np.array(Y).astype(int).reshape(-1)]
                Z1 = np.dot(X, self.W1) + self.b1
                A1 = self.sigmoid(Z1)
                Z2 = np.dot(A1, self.W2) + self.b2
                A2 = self.softmax(Z2)
                dZ2 = A2 - Y
                dZ2_dW2 = A1
                dW2 = np.dot(dZ2_dW2.T, dZ2) / len(X) + np.multiply(regularizer, self.W2)/len(X)
                db2 = dZ2 / len(X)
                dW1 = (np.dot(X.T, self.sigmoid_der(Z1) * np.dot(dZ2, self.W2.T))) / len(X) + np.multiply(regularizer,
                                                                                                          self.W1)/len(X)
                db1 = (np.dot(dZ2, self.W2.T) * self.sigmoid_der(Z1)) / len(X)
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1.sum(axis=0)
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2.sum(axis=0)
                j = self.cost_function(A2, Y)
                avg_l += j
                avg_a += self.accuracy(A2, Y)
            self.train_accuracy.append(avg_a / len(batch))
            self.train_error.append(avg_l / len(batch))
            Z1 = np.dot(X_test, self.W1) + self.b1
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.softmax(Z2)
            self.test_error.append(self.cost_function(A2, Y_test))
            self.test_accuracy.append(self.accuracy(A2, Y_test))


def main():
    model = MLP(200)
    model.mini_batch_train()
    print("Highest Accuracy in Train")
    print(max(model.train_accuracy))
    print("Highest Accuracy in Test")
    print(max(model.test_accuracy))
    plt.plot(model.train_error, label="Train Error")
    plt.plot(model.test_error, label="Test Error")
    plt.plot(model.train_accuracy, label="Train Accuracy")
    plt.plot(model.test_accuracy, label="Test Accuracy")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    main()
