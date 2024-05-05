from DBSCAN import DBSCAN
from util import get_data, visualization, data_visualization
import matplotlib.pyplot as plt  # 确保导入了matplotlib.pyplot
import time
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score


def show_dataset():
    X, y = get_data()
    data_visualization(X, y)
    plt.show()


def show_dbscan():
    X, y = get_data()
    X = X[:, [2, 3]]

    dbscan = DBSCAN(eps=0.3, min_samples=5)
    start_time = time.time()
    dbscan.fit(X)
    print("Time cost: ", time.time() - start_time)
    print(dbscan.labels_)
    silhouette = silhouette_score(X, dbscan.labels_)
    db_index = davies_bouldin_score(X, dbscan.labels_)

    print("Silhouette Coefficient: ", silhouette)
    print("DB Index: ", db_index)
    visualization(X, dbscan)
    plt.show()


if __name__ == "__main__":
    # show_dataset()
    show_dbscan()
