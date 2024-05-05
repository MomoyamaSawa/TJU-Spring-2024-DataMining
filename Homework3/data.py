from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np


def get_data():
    iris = fetch_ucirepo(id=53)
    X = iris.data.features
    y = iris.data.targets
    N_CLUSTERS = len(np.unique(y))  # 获取唯一值的数量
    return X, y, N_CLUSTERS


if __name__ == "__main__":
    """ "
    观察数据集原本的聚类信息
    """
    X, y, _ = get_data()

    # create a dataframe with the features and the target
    df = X.copy()
    df["target"] = y

    # create a color map for the targets
    colors = {
        "Iris-setosa": "red",
        "Iris-versicolor": "green",
        "Iris-virginica": "blue",
    }

    # create a scatter matrix
    scatter_matrix(
        df[df.columns[:-1]],
        figsize=(10, 10),
        c=df["target"].apply(lambda x: colors[x]),
        alpha=0.8,
    )

    plt.show()
