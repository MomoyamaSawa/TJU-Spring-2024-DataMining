import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from scipy.cluster.hierarchy import dendrogram, fcluster, optimal_leaf_ordering
from scipy.spatial.distance import pdist
from sklearn import metrics
from agglomerative import LinkageEnum


def plot_clusters(data, labels, n_clusters):
    df = data.copy()
    df["target"] = labels

    # create a color map for the targets
    cmap = plt.cm.get_cmap("rainbow", n_clusters)  # get a colormap
    colors = {
        i: cmap(i) for i in range(n_clusters)
    }  # create a color map for the targets

    # create a scatter matrix
    scatter_matrix(
        df[df.columns[:-1]],
        figsize=(10, 10),
        c=df["target"].apply(lambda x: colors[x - 1]),
        alpha=0.8,
    )


def plot_dendrogram(matrix, n_clusters):

    # 创建一个新的figure
    plt.figure(figsize=(10, 10))

    # 使用Z和clusters绘制树状图
    dendrogram(matrix, color_threshold=matrix[-(n_clusters - 1), 2])


def get_cluster_labels(linkage_matrix, n_clusters):
    # 使用fcluster函数获取聚类标签
    labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
    return labels


if __name__ == "__main__":
    from data import get_data
    import time

    X, y, N_CLUSTERS = get_data()

    # 记录开始时间
    start_time = time.time()
    # 使用你的聚类函数
    matrix = LinkageEnum.SINGLE(X)
    # 记录结束时间
    end_time = time.time()
    # 计算并打印运行时间
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # 优化聚类结果
    matrix = optimal_leaf_ordering(matrix, pdist(X))

    # 获取聚类标签
    labels = get_cluster_labels(matrix, N_CLUSTERS)

    # 计算内部指标
    silhouette_score = metrics.silhouette_score(X, labels)
    davies_bouldin_score = metrics.davies_bouldin_score(X, labels)

    # 计算外部指标
    adjusted_rand_score = metrics.adjusted_rand_score(y.values.ravel(), labels)
    mutual_info_score = metrics.adjusted_mutual_info_score(y.values.ravel(), labels)

    print(f"Silhouette Coefficient: {silhouette_score}")
    print(f"Davies-Bouldin Index: {davies_bouldin_score}")
    print(f"Adjusted Rand Index: {adjusted_rand_score}")
    print(f"Adjusted Mutual Information: {mutual_info_score}")

    # 绘制聚类结果
    plot_clusters(X, labels, N_CLUSTERS)

    # 绘制层级图
    plot_dendrogram(matrix, N_CLUSTERS)

    plt.show()
