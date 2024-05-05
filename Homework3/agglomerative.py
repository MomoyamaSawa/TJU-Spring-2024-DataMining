from enum import Enum
import math
import numpy as np


def data_pre(data):
    data = data.values
    n = data.shape[0]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.sqrt(np.sum((data[i] - data[j]) ** 2)))

    # 计算观察值的数量
    num_obs = len(data)

    return np.array(dists), num_obs


def condensed_index(n, i, j):
    """
    Calculate the condensed index of element (i, j) in an n x n condensed
    matrix.
    """
    if i < j:
        return n * i - (i * (i + 1) // 2) + (j - i - 1)
    elif i > j:
        return n * j - (j * (j + 1) // 2) + (i - j - 1)


class LinkageUnionFind:
    """Structure for fast cluster labeling in unsorted dendrogram."""

    def __init__(self, n):
        self.parent = np.arange(2 * n - 1, dtype=int)
        self.next_label = n
        self.size = np.ones(2 * n - 1, dtype=int)

    def merge(self, x, y):
        self.parent[x] = self.next_label
        self.parent[y] = self.next_label
        size = self.size[x] + self.size[y]
        self.size[self.next_label] = size
        self.next_label += 1
        return size

    def find(self, x):
        p = x

        while self.parent[x] != x:
            x = self.parent[x]

        while self.parent[p] != x:
            p, self.parent[p] = self.parent[p], x

        return x


def label(Z, n):
    """Correctly label clusters in unsorted dendrogram."""
    uf = LinkageUnionFind(n)

    for i in range(n - 1):
        x, y = int(Z[i, 0]), int(Z[i, 1])
        x_root, y_root = uf.find(x), uf.find(y)
        if x_root < y_root:
            Z[i, 0], Z[i, 1] = x_root, y_root
        else:
            Z[i, 0], Z[i, 1] = y_root, x_root
        Z[i, 3] = uf.merge(x_root, y_root)


def _ward(d_xi, d_yi, d_xy, size_x, size_y, size_i):
    t = 1.0 / (size_x + size_y + size_i)
    return math.sqrt(
        (size_i + size_x) * t * d_xi * d_xi
        + (size_i + size_y) * t * d_yi * d_yi
        - size_i * t * d_xy * d_xy
    )


def _complex(d_xi, d_yi, d_xy, size_x, size_y, size_i):
    return max(d_xi, d_yi)


def single_linkage(X):
    """Perform hierarchy clustering using MST algorithm for single linkage.

    Parameters
    ----------
    dists : ndarray
        A condensed matrix stores the pairwise distances of the observations.
    n : int
        The number of observations.

    Returns
    -------
    Z : ndarray, shape (n - 1, 4)
        Computed linkage matrix.
    """

    dists, n = data_pre(X)

    Z_arr = np.empty((n - 1, 4))

    # Which nodes were already merged.
    merged = np.zeros(n, dtype=int)

    D = np.empty(n)
    D[:] = float("inf")

    x, y = 0, 0
    current_min = float("inf")

    for k in range(n - 1):
        current_min = float("inf")
        merged[x] = 1
        for i in range(n):
            if merged[i] == 1:
                continue

            dist = dists[condensed_index(n, x, i)]
            if D[i] > dist:
                D[i] = dist

            if D[i] < current_min:
                y = i
                current_min = D[i]

        Z_arr[k, 0] = x
        Z_arr[k, 1] = y
        Z_arr[k, 2] = current_min
        x = y

    # Sort Z by cluster distances.
    order = np.argsort(Z_arr[:, 2], kind="mergesort")
    Z_arr = Z_arr[order]

    # Find correct cluster labels and compute cluster sizes inplace.
    label(Z_arr, n)

    return Z_arr


def ward_linkage(X):
    dists, n = data_pre(X)
    return nn_chain(dists, n, _ward)


def complex_linkage(X):
    dists, n = data_pre(X)
    return nn_chain(dists, n, _complex)


def nn_chain(dists, n, new_dist):
    Z_arr = np.empty((n - 1, 4))

    D = dists.copy()  # Distances between clusters.
    size = np.ones(n, dtype=int)  # Sizes of clusters.

    # Variables to store neighbors chain.
    cluster_chain = np.empty(n, dtype=int)
    chain_length = 0

    for k in range(n - 1):
        if chain_length == 0:
            chain_length = 1
            for i in range(n):
                if size[i] > 0:
                    cluster_chain[0] = i
                    break

        # Go through chain of neighbors until two mutual neighbors are found.
        while True:
            x = cluster_chain[chain_length - 1]

            # We want to prefer the previous element in the chain as the
            # minimum, to avoid potentially going in cycles.
            if chain_length > 1:
                y = cluster_chain[chain_length - 2]
                current_min = D[condensed_index(n, x, y)]
            else:
                current_min = np.inf

            for i in range(n):
                if size[i] == 0 or x == i:
                    continue

                dist = D[condensed_index(n, x, i)]
                if dist < current_min:
                    current_min = dist
                    y = i

            if chain_length > 1 and y == cluster_chain[chain_length - 2]:
                break

            cluster_chain[chain_length] = y
            chain_length += 1

        # Merge clusters x and y and pop them from stack.
        chain_length -= 2

        # This is a convention used in fastcluster.
        if x > y:
            x, y = y, x

        # get the original numbers of points in clusters x and y
        nx = size[x]
        ny = size[y]

        # Record the new node.
        Z_arr[k, 0] = x
        Z_arr[k, 1] = y
        Z_arr[k, 2] = current_min
        Z_arr[k, 3] = nx + ny
        size[x] = 0  # Cluster x will be dropped.
        size[y] = nx + ny  # Cluster y will be replaced with the new cluster

        # Update the distance matrix.
        for i in range(n):
            ni = size[i]
            if ni == 0 or i == y:
                continue

            D[condensed_index(n, i, y)] = new_dist(
                D[condensed_index(n, i, x)],
                D[condensed_index(n, i, y)],
                current_min,
                nx,
                ny,
                ni,
            )

    # Sort Z by cluster distances.
    order = np.argsort(Z_arr[:, 2], kind="mergesort")
    Z_arr = Z_arr[order]

    # Correct the cluster labels.
    label(Z_arr, n)

    return Z_arr


class LinkageEnum(Enum):
    SINGLE = single_linkage
    WARD = ward_linkage
    COMPLEX = complex_linkage
