import numpy as np
import matplotlib.pyplot as plt


class MaximumEntropyClassifier:
    def __init__(self):
        self.W = None
        self.b = None

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

    def mini_batch_train(self, epochs, learning_rate = 0.01):
        with open('iris_train.dat', 'r') as f:
            data = [i.strip().split(",") for i in f.readlines()]
        data = np.array(data).astype(float)
        with open('iris_test.dat', 'r') as f:
            test_data = [i.strip().split(",") for i in f.readlines()]
        test_data = np.array(test_data).astype(float)
        X_test = test_data[:, :-1]
        Y_test = test_data[:, -1]
        Y_test = np.eye(3)[np.array(Y_test).astype(int).reshape(-1)]
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []
        N, D = data[:, :-1].shape
        k = len(np.unique(data[:, -1]))
        self.W = np.random.randn(D, k)
        self.b = np.random.randn(k)
        for i in range(epochs):
            avg_loss = 0
            avg_accuracy = 0
            np.random.shuffle(data)
            batch = np.split(data, 5)
            for b in batch:
                X = b[:, :-1]
                Y = b[:, -1]
                Y_hotencode = np.eye(3)[np.array(Y).astype(int).reshape(-1)]
                p = self.feed_forward(X, self.W, self.b)
                j = self.cost_function(p, Y_hotencode, self.W)
                dw, db = self.gradient(X, Y_hotencode, p)
                self.W -= learning_rate * dw
                self.b -= learning_rate * db
                avg_loss += j
                avg_accuracy += self.calc_accuracy(p, Y_hotencode)
            self.train_loss.append(avg_loss / len(batch))
            self.train_accuracy.append(avg_accuracy / len(batch))
            p = self.feed_forward(X_test, self.W, self.b)
            loss = self.cost_function(p, Y_test, self.W)
            self.test_loss.append(loss)
            accuracy = self.calc_accuracy(p, Y_test)
            self.test_accuracy.append(accuracy)
        print("Test Accuracy after all the Epochs", accuracy)
        print("Train Accuracy after all the Epochs", (avg_accuracy) / len(batch))
        print("Test Loss after all the Epochs", loss)
        print("Train Loss after all the Epochs", avg_loss / len(batch))


    def calc_accuracy(self,p, y):
        accuracy = 0
        for i in range(len(p)):
            if np.argmax(p[i]) == np.argmax(y[i]):
                accuracy += 1
        return accuracy / len(p)


def main():
    model = MaximumEntropyClassifier()
    model.mini_batch_train(500)
    plt.plot(model.test_accuracy, label="Test Accuracy")
    plt.plot(model.train_accuracy, label="Train Accuracy")
    plt.plot(model.test_loss, label="Test Loss")
    plt.plot(model.train_loss, label="Train Loss")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    main()
