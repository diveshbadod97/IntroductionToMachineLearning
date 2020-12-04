import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LogisticRegressionSinglePerceptron:
    def __init__(self, lr, num_iter):
        self.lr = lr
        self.num_iter = num_iter

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_function(self, h, y):
        return ((-y * np.log(h)) - ((1 - y) * np.log(1 - h))).mean()

    def train(self, X, Y):
        self.error_rate = []
        self.weights = np.random.randn(X.shape[1])
        for i in range(self.num_iter):
            z = np.dot(X, self.weights)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - Y)) / Y.size
            self.weights -= self.lr * gradient
            z = np.dot(X, self.weights)
            h = self.sigmoid(z)
            loss = self.cost_function(h, Y)
            self.error_rate.append(loss)

    def predict(self, X):
        return self.sigmoid(np.dot(X, self.weights)).round()


def main():
    l = ["Frogs.csv", "Frogs-subsample.csv"]
    for i in l:
        df = pd.read_csv(i).to_numpy()
        X = df[:, :2]
        Y = []
        X = np.array(X).astype(float)
        for i in df:
            if i[2] == "HylaMinuta":
                Y.append(0)
            else:
                Y.append(1)
        Y = np.array(Y)
        model = LogisticRegressionSinglePerceptron(lr=0.1, num_iter=15000)
        model.train(X, Y)
        c1 = df[df[:, 2] == 'HylaMinuta']
        c2 = df[df[:, 2] == 'HypsiboasCinerascens']
        plt.scatter(c1[:, 0], c1[:, 1], c="red", label='HylaMinuta')
        plt.scatter(c2[:, 0], c2[:, 1], c="green", label='HypsiboasCinerascens')
        x = np.arange(-0.5, 0.5, 0.1)
        print(model.weights)
        """Uncomment to predict the values and print accuracy"""
        #prediction = model.predict(X)
        #print(np.mean(prediction == Y) * 100)
        plt.plot(x, (-model.weights[0] / model.weights[1]) * x, color='black', label='Decision Boundary')
        plt.title("Scatter Plot")
        plt.legend(loc="upper right")
        plt.xlabel("MFCCs_10")
        plt.ylabel("MFCCs_17")
        plt.show()
        """Uncomment to Plot Error rate graph"""
        # plt.plot(model.error_rate)
        # plt.xlabel("Number of iterations")
        # plt.ylabel("Loss")
        # plt.show()


if __name__ == '__main__':
    main()
