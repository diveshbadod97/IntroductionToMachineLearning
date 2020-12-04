import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def scatter_plot(df1):
    c1 = df1[df1[:, 2] == 'HylaMinuta']
    c2 = df1[df1[:, 2] == 'HypsiboasCinerascens']
    plt.scatter(c1[:, 0], c1[:, 1], c="red", label='HylaMinuta')
    plt.scatter(c2[:, 0], c2[:, 1], c="green", label='HypsiboasCinerascens')
    plt.title("Scatter Plot")
    plt.legend(loc="upper right")
    plt.xlabel("MFCCs_10")
    plt.ylabel("MFCCs_17")
    plt.show()


def histogram_plot(df1):
    MFCCs_10 = [i[0] for i in df1]
    MFCCs_17 = [i[1] for i in df1]
    plt.hist(MFCCs_10, bins=20)
    plt.title("Histogram for feature MFCCs_10")
    plt.xlabel("Frequency")
    plt.ylabel("Data Values")
    plt.show()
    plt.hist(MFCCs_17, bins=20)
    plt.title("Histogram for feature MFCCs_17")
    plt.xlabel("Data Values")
    plt.ylabel("Frequency")
    plt.show()


def line_plot(df1):
    MFCCs_10 = [i[0] for i in df1]
    MFCCs_17 = [i[1] for i in df1]
    MFCCs_10.sort()
    MFCCs_17.sort()
    plt.plot(MFCCs_10)
    plt.title("Line Graph for feature MFCCs_10")
    plt.ylabel("Data Values")
    plt.show()
    plt.plot(MFCCs_17)
    plt.title("Line Graph for feature MFCCs_17")
    plt.ylabel("Data Values")
    plt.show()


def box_plot(df1):
    c1 = df1[df1[:, 2] == 'HylaMinuta']
    c2 = df1[df1[:, 2] == 'HypsiboasCinerascens']
    plt.boxplot(c1[:, :2])
    plt.title("Boxplot for HylaMinuta")
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.show()
    plt.boxplot(c2[:, :2])
    plt.title("Boxplot for HypsiboasCinerascens")
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.show()


def error_bar_graph(df1):
    f1 = [i[0] for i in df1 if i[2] == 'HylaMinuta']
    f2 = [i[1] for i in df1 if i[2] == 'HylaMinuta']
    f3 = [i[0] for i in df1 if i[2] == 'HypsiboasCinerascens']
    f4 = [i[1] for i in df1 if i[2] == 'HypsiboasCinerascens']
    f1 = np.array(f1)
    f2 = np.array(f2)
    f3 = np.array(f3)
    f4 = np.array(f4)
    x_pos = np.arange(4)
    CTEs = [np.mean(f1), np.mean(f2), np.mean(f3), np.mean(f4)]
    error = [np.std(f1), np.std(f2), np.std(f3), np.std(f4)]
    plt.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.tight_layout()
    plt.title("Bar graph with error bars")
    plt.show()


def raw_features(file_name):
    df = pd.read_csv(file_name, delimiter=",")
    df1 = df.to_numpy()
    scatter_plot(df1)
    histogram_plot(df1)
    line_plot(df1)


def feature_distributions(file_name):
    df = pd.read_csv(file_name, delimiter=",")
    df1 = df.to_numpy()
    box_plot(df1)
    error_bar_graph(df1)


def descriptive_statistics(file_name):
    df = pd.read_csv(file_name, delimiter=",")
    df1 = df.to_numpy()
    MFCCs_10 = [i[0] for i in df1]
    MFCCs_17 = [i[1] for i in df1]
    MFCCs_10_1 = [i[0] for i in df1 if i[2] == "HylaMinuta"]
    MFCCs_10_2 = [i[0] for i in df1 if i[2] == "HypsiboasCinerascens"]
    MFCCs_17_1 = [i[1] for i in df1 if i[2] == "HylaMinuta"]
    MFCCs_17_2 = [i[1] for i in df1 if i[2] == "HypsiboasCinerascens"]
    MFCCs_10 = np.array(MFCCs_10)
    MFCCs_17 = np.array(MFCCs_17)
    MFCCs_10_1 = np.array(MFCCs_10_1)
    MFCCs_10_2 = np.array(MFCCs_10_2)
    MFCCs_17_1 = np.array(MFCCs_17_1)
    MFCCs_17_2 = np.array(MFCCs_17_2)
    MFCCs_10_mean = np.mean(MFCCs_10)
    MFCCs_17_mean = np.mean(MFCCs_17)
    MFCCs_10_1mean = np.mean(MFCCs_10_1)
    MFCCs_17_1mean = np.mean(MFCCs_17_1)
    MFCCs_10_2mean = np.mean(MFCCs_10_2)
    MFCCs_17_2mean = np.mean(MFCCs_17_2)
    print("Mean for feature MFCCs_10:", MFCCs_10_mean)
    print("Mean for feature MFCCs_17:", MFCCs_17_mean)
    print("Mean for feature MFCCs_10 class 1:", MFCCs_10_1mean)
    print("Mean for feature MFCCs_17 class 1:", MFCCs_17_1mean)
    print("Mean for feature MFCCs_10 class 2:", MFCCs_10_2mean)
    print("Mean for feature MFCCs_17 class 2:", MFCCs_17_2mean)
    data = [[i[0], i[1]] for i in df1]
    data = np.array(data)
    cov_matrix = np.cov(data.T)
    data1 = [[i[0], i[1]] for i in df1 if i[2] == "HylaMinuta"]
    data2 = [[i[0], i[1]] for i in df1 if i[2] == "HypsiboasCinerascens"]
    cov_matrix1 = np.cov(np.array(data1).T)
    cov_matrix2 = np.cov(np.array(data2).T)
    print("Covariance Matrix:", cov_matrix)
    print("Covariance Matrix class 1:", cov_matrix1)
    print("Covariance Matrix class 2:", cov_matrix2)
    print("Standard Deviation for feature MFCCs_10:", np.std(MFCCs_10))
    print("Standard Deviation for feature MFCCs_17:", np.std(MFCCs_17))
    print("Standard Deviation for feature MFCCs_10_1 class 1:", np.std(MFCCs_10_1))
    print("Standard Deviation for feature MFCCs_17_1 class 1:", np.std(MFCCs_17_1))
    print("Standard Deviation for feature MFCCs_10_2 class 2:", np.std(MFCCs_10_2))
    print("Standard Deviation for feature MFCCs_17_2 class 2:", np.std(MFCCs_17_2))


def main():
    l = ["Frogs.csv", "Frogs-subsample.csv"]
    for i in l:
        raw_features(i)
        feature_distributions(i)
        descriptive_statistics(i)


if __name__ == '__main__':
    main()
