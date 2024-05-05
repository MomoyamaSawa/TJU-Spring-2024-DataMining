""""
用scipy库检查自己实现的结果是否正确
"""

from enum import Enum
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from start import plot_clusters, plot_dendrogram, get_cluster_labels
from data import get_data


def generate_linkage_matrix_single(data):
    # 使用单链接方法生成链接矩阵
    linkage_matrix = linkage(data, method="single")
    return linkage_matrix


def generate_linkage_matrix_complete(data):
    # 使用全链接方法生成链接矩阵
    linkage_matrix = linkage(data, method="complete")
    return linkage_matrix


def generate_linkage_matrix_ward(data):
    # 使用ward方法生成链接矩阵
    linkage_matrix = linkage(data, method="ward")
    return linkage_matrix


class MethodEnum(Enum):
    """方法枚举"""

    SINGLE = generate_linkage_matrix_single
    COMPLETE = generate_linkage_matrix_complete
    WARD = generate_linkage_matrix_ward


if __name__ == "__main__":
    # 获取数据
    data, y, n_clusters = get_data()
    # 生成链接矩阵
    linkage_matrix = MethodEnum.WARD(data)
    # 获取聚类标签
    labels = get_cluster_labels(linkage_matrix, n_clusters)
    # 绘制层级图
    plot_dendrogram(linkage_matrix, n_clusters)
    plot_clusters(data, labels, n_clusters)
    plt.show()
