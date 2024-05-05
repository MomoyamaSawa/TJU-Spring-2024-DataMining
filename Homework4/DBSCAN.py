from scipy.spatial import KDTree
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        self.core_samples_indices_ = []
        self.labels_ = -np.ones(len(X), dtype=int)
        cluster_id = 0
        tree = KDTree(X)
        all_neighbors = tree.query_ball_point(X, self.eps)  # 批量查询

        # 定义一个处理每个点的函数
        def process_point(i):
            if self.labels_[i] != -1:
                return

            neighbors = all_neighbors[i]
            if len(neighbors) < self.min_samples:
                self.labels_[i] = 0  # Mark as noise
                return

            nonlocal cluster_id
            cluster_id += 1  # Start a new cluster
            self.labels_[i] = cluster_id
            self.core_samples_indices_.append(i)

            seeds = set(neighbors)
            seeds.remove(i)

            while seeds:
                j = seeds.pop()
                if self.labels_[j] == 0:
                    self.labels_[j] = cluster_id  # Change noise to border point
                if self.labels_[j] != -1:
                    continue  # Already processed

                self.labels_[j] = cluster_id
                neighbors = all_neighbors[j]
                if len(neighbors) >= self.min_samples:
                    seeds.update(neighbors)

        # 使用线程池并行处理每个点
        with ThreadPoolExecutor() as executor:
            executor.map(process_point, range(len(X)))

        return self


if __name__ == "__main__":
    from util import get_test_data, visualization

    # 获取测试数据
    X, y = get_test_data()

    # 初始化DBSCAN对象
    dbscan = DBSCAN(eps=0.1, min_samples=5)

    # 对数据集进行聚类
    dbscan.fit(X)

    # 输出聚类结果
    print(dbscan.labels_)

    # 可视化聚类结果
    visualization(X, dbscan)
