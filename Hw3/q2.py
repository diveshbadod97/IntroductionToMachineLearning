import numpy as np
import pandas as pd


def maximum_likelihood(features, data):
    prob = {}
    data1 = data[:, -1]
    t = []
    f = []
    for i in data1:
        if i:
            t.append(i)
        else:
            f.append(i)
    true_prob = len(t) / len(data)
    false_prob = len(f) / len(data)
    for i, fe in enumerate(features[: len(features) - 1]):
        if i == 6 or i == 7:
            mean_T, std_T = calc_mean_and_std(data, True, i)
            mean_F, std_F = calc_mean_and_std(data, False, i)
            prob[fe] = [[mean_T, std_T], [mean_F, std_F]]
        else:
            prob[fe] = confusion_matrix(data, t, f, i)
    return true_prob, false_prob, prob


def confusion_matrix(data, t, f, i):
    TT = []
    TF = []
    FT = []
    FF = []
    for j in range(len(data[:, i])):
        if data[:, i][j] == True and data[:, -1][j] == True:
            TT.append(j)
        if data[:, i][j] == True and data[:, -1][j] == False:
            TF.append(j)
        if data[:, i][j] == False and data[:, -1][j] == True:
            FT.append(j)
        if data[:, i][j] == False and data[:, -1][j] == False:
            FF.append(j)
    return [[len(TT) / len(t), len(TF) / len(f)], [len(FT) / len(t), len(FF) / len(f)]]


def calc_mean_and_std(data, val, i):
    mean = np.mean(data[data[:, -1] == val, i])
    std = np.var(data[data[:, -1] == val, i])
    return mean, std


def predict(t_prob, f_prob, prob, data, features):
    out = 0
    t_pred = t_prob
    f_pred = f_prob
    for i, fe in enumerate(features[: len(features) - 1]):
        if i == 6 or i == 7:
            t_pred = normal_dist_eval(data[i], prob[fe][0][0], prob[fe][0][1])
            f_pred = normal_dist_eval(data[i], prob[fe][1][0], prob[fe][1][1])
        else:
            if data[i]:
                t_pred *= prob[fe][0][0]
                f_pred *= prob[fe][0][1]
            else:
                t_pred *= prob[fe][1][0]
                f_pred *= prob[fe][1][1]
    t_pred /= (t_pred + f_pred)
    f_pred /= (t_pred + f_pred)
    if t_pred > 0.5:
        if data[-1]:
            out += 1
    else:
        if not data[-1]:
            out += 1
    return out


def normal_dist_eval(val, mean, var):
    return 1 / (np.sqrt(2 * np.pi * var)) * np.exp((-(val - mean) ** 2) / (2 * var))


def main():
    data = pd.read_csv('q3.csv')
    features = data.columns
    data = data.to_numpy()
    features = features.to_numpy()
    t_prob, f_prob, prob = maximum_likelihood(features, data)
    bool = True
    p = 1
    for i, j in prob.items():
        print(str(i) + "        True                False")
        for k in j:
            p += 1
            if p % 2 == 0:
                print("Spam True", end=" ")
            else:
                print("Spam False", end=" ")
            print(k)
        print("##########################################################")
    data1 = pd.read_csv("q3b.csv")
    features1 = data1.columns
    data1 = data1.to_numpy()
    accuracy = 0
    for i in range(len(data1)):
        accuracy += predict(t_prob, f_prob, prob, data1[i], features1)
    accuracy /= len(data1)
    print("Accuracy", accuracy * 100)
    print("Loss", (1 - accuracy) * 100)
    c = np.array([[5, 4, 6, 2], [4, 6, 3], [0, 4, 5], [0], [6]])
    subset_feature_accuracy = []
    for i in c:
        accuracy1 = 0
        t_prob1, f_prob1, prob1 = maximum_likelihood(features[i], data[:, i])
        for j in range(len(data1)):
            accuracy1 += predict(t_prob1, f_prob1, prob1, data1[j], features1[i])
        accuracy1 /= len(data1)
        subset_feature_accuracy.append(accuracy1)
    print(subset_feature_accuracy)
    print("Maximum Accuracy Obtained after randomly ignoring few features", end=" ")
    print(max(subset_feature_accuracy)*100)


if __name__ == '__main__':
    main()
