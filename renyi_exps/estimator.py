from scipy.spatial import KDTree
import numpy as np
from sklearn.neighbors import NearestNeighbors

def estimate_rynei (X,Y,  alpha=2,k=3,rounding=5):
    X = np.asarray(X)
    Y = np.asarray(Y)
    # print(X.shape)
    d = X.shape[1]
    n = X.shape[0]
    m = Y.shape[0]

    if rounding == None:
        X_nodups, X_counts = np.unique(X, axis=0, return_counts=True)
        Y_nodups, Y_counts = np.unique(Y, axis=0, return_counts=True)
    else:
        # use round() to crudely merge nearby points
        X_nodups, X_counts = np.unique(X.round(decimals=rounding), axis=0, return_counts=True)
        Y_nodups, Y_counts = np.unique(Y.round(decimals=rounding), axis=0, return_counts=True)

    X_tree = KDTree(X_nodups)
    Y_tree = KDTree(Y_nodups)

    k_p = [];
    k_q = [];
    rho = [];
    nu = []
    for y in Y_nodups:
        dist, indices = Y_tree.query(y, k + 1)
        k_q.append(np.sum(Y_counts[indices]))
        nu.append(np.max(dist))  # max distance
        max_dist = np.max(dist)
        indicies_x_point = X_tree.query_ball_point(y, max_dist)
        k_p.append(np.sum(X_counts[indicies_x_point]))
        rho.append(max_dist)
    r = 0
    kp_sum = sum(k_p); kq_sum = sum(k_q)
    for i in range(len(Y_nodups)):
        p_density = (k_p[i] / kp_sum) * (1 / ((rho[i]) ** d))
        q_density = (k_q[i] / kq_sum) * (1 / ((nu[i]) ** d))
        ratio = (p_density / q_density) ** alpha
        r += ratio * k_q[i] / (kq_sum)

    value = (1 / (alpha - 1)) * np.log(r)
    # Ensure the result is non-negative
    value = max(0, value)

    return value
