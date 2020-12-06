import mnist
import numpy as np
import matplotlib.pyplot as plt


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

    def cost_function(self, p, Y, regularizer=0.3):
        return -1 * (np.sum(Y * np.log(p)) / len(Y)) + (
                (regularizer / 2) * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))) / len(Y)

    def train(self, X_train, Y_train, learning_rate=0.001, regularizer=0.3):
        mini = 20000
        partition = int(len(X_train) * 0.8)
        X_val = X_train[-partition:]
        X_train = X_train[partition:]
        Y_val = Y_train[-partition:]
        Y_train = Y_train[partition:]
        self.W1 = np.random.randn(X_train.shape[1], 500) * 0.01
        self.b1 = np.zeros(500)
        self.W2 = np.random.randn(500, 10) * 0.01
        self.b2 = np.zeros(10)
        self.train_error = []
        self.val_error = []
        for i in range(self.num_iter):
            batchX = np.split(X_train, 5)
            batchY = np.split(Y_train, 5)
            avg_l = 0
            i = 0
            for b in batchX:
                Z1 = np.dot(b, self.W1) + self.b1
                A1 = self.sigmoid(Z1)
                Z2 = np.dot(A1, self.W2) + self.b2
                A2 = self.softmax(Z2)
                dZ2 = A2 - batchY[i]
                dZ2_dW2 = A1
                dW2 = np.dot(dZ2_dW2.T, dZ2) / len(b) + np.multiply(regularizer, self.W2) / len(b)
                db2 = dZ2 / len(b)
                dW1 = (np.dot(b.T, self.sigmoid_der(Z1) * np.dot(dZ2, self.W2.T))) / len(b) + np.multiply(regularizer,
                                                                                                          self.W1) / len(
                    b)
                db1 = (np.dot(dZ2, self.W2.T) * self.sigmoid_der(Z1)) / len(b)
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1.sum(axis=0)
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2.sum(axis=0)
                j = self.cost_function(A2, batchY[i])
                i += 1
                avg_l += j
            self.train_error.append(avg_l / len(batchX))
            Z1 = np.dot(X_val, self.W1) + self.b1
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.softmax(Z2)
            val_e = self.cost_function(A2, Y_val)
            if val_e < mini:
                mini = val_e
                self.val_error.append(mini)
            elif val_e > mini:
                break

    def test(self, x_test, y_test):
        Z1 = np.dot(x_test, self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)
        accuracy = self.accuracy(A2, y_test)
        print(accuracy)

    def accuracy(self, p, y):
        self.accurate_class = [0] * 10
        self.total = [0] * 10
        accuracy = 0
        for i in range(len(p)):
            if np.argmax(p[i]) == np.argmax(y[i]):
                self.accurate_class[np.argmax(y[i])] += 1
                accuracy += 1
            self.total[np.argmax(y[i])] += 1
        print(self.total)
        print(self.accurate_class)
        plt.hist(self.accurate_class)
        plt.show()
        # plt.hist(p)
        # plt.show()
        return 100 * accuracy / len(p)


def main():
    x_train = mnist.train_images()
    y_train = mnist.train_labels()
    x_test = mnist.test_images()
    y_test = mnist.test_labels()
    x_train = np.asarray(x_train).astype(np.float32) / 255
    y_train = np.asarray(y_train).astype(np.int32)
    x_test = np.asarray(x_test).astype(np.float32) / 255
    y_test = np.asarray(y_test).astype(np.int32)
    y_train = np.eye(10)[np.array(y_train).astype(int).reshape(-1)]
    y_test = np.eye(10)[np.array(y_test).astype(int).reshape(-1)]
    model = MLP(100)
    # print(x_train.shape)
    # print(y_train.shape)
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    model.train(x_train, y_train)
    model.test(x_test, y_test)
    plt.plot(model.val_error, label="Validation Error")
    plt.plot(model.train_error, label="Train Error")
    plt.legend(loc = 'upper right')
    plt.show()


if __name__ == '__main__':
    main()
