from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns


def get_test_data():
    # 创建一个半月形的数据集
    return make_moons(n_samples=500, noise=0.05)


def visualization(X, dbscan):
    # 可视化结果
    plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_)
    plt.title("DBSCAN Clustering")


def get_data():
    # iris数据集
    iris = fetch_ucirepo(id=53)
    X = iris.data.features.values
    y = iris.data.targets.values.ravel()
    return X, y


def data_visualization(X, y):
    # 假设X和y已经被定义并且加载了数据
    iris_df = pd.DataFrame(
        X, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
    iris_df["species"] = y

    sns.pairplot(iris_df, hue="species")
