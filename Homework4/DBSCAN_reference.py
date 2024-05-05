from sklearn.cluster import DBSCAN
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
