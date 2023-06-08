

def exhaust_partition(X,clustn):
    import itertools
    import numpy as np
    best_pairs = []
    for pair in itertools.combinations(range(X.shape[1]),clustn):
        pair = list(pair)
        cost = X[:,pair].min(1).sum()
        cfg_which = np.argmin(X[:,pair],axis=1)
        best_pairs.append((cost,pair,cfg_which))
    best_res = sorted(best_pairs)[0]
    best_cfg_i = best_res[1]
    c_labels = best_res[2]
    return best_cfg_i, c_labels


def optimize_partition(X,clustn):
    import scipy.optimize
    import numpy as np

    bigM = np.prod(X.shape)

    N_v = X.shape[1]
    A = []
    for i in range(X.shape[0]):
        v = np.zeros_like(X)
        v[i] = 1
        A.append(list(v.ravel()) + list(np.zeros(N_v)))
    A2 = []
    b2 = []
    for i in range(X.shape[1]):
        v = np.zeros_like(X)
        v[:,i] = 1
        v2 = np.zeros(N_v)
        v2[i] = -bigM
        A2.append(list(v.ravel()) + list(v2))
        b2.append(0)

    A2.append(list(np.zeros_like(X).ravel()) + list(np.ones(N_v)))

    b2.append(clustn)

    A = np.array(A)
    b = np.ones(X.shape[0])

    res = scipy.optimize.linprog(
        list(X.ravel()) + list(np.zeros(N_v)),
        A_eq=A, b_eq=b,
        A_ub=A2, b_ub=b2,
        bounds=[(0, 1) for _ in range(len(X.ravel()) + N_v)],
        integrality=[1 for _ in range(len(X.ravel()) + N_v)],
        options={"disp": False})
    res_x = res.x[:np.prod(X.shape)].reshape((X.shape))

    best_cfg_i = list(np.where(res_x.sum(0) >0)[0])

    return best_cfg_i, np.argmin(X[:,best_cfg_i],axis=1)

def cluster_partition(X,K,n_init=50):
    import sklearn.cluster
    import numpy as np

    y = (X-X.mean(1,keepdims=True))/X.std(1,keepdims=True)
    clf = sklearn.cluster.KMeans(K,n_init=n_init)
    clf.fit(y)
    best_cfg_i = []

    for i in range(K):
        idx = clf.labels_ == i
        vec_scores = X[idx].sum(0)
        best_vec_config = np.argsort(vec_scores)
        best_cfg_i.append(best_vec_config[0])

    c_labels=clf.labels_
    return best_cfg_i, c_labels

def greedy_partition(X,K):
    import numpy as np
    best_cfg_i = [np.argmin(X.sum(axis=0))]
    for k in range(K-1):
        cost_basis = np.minimum(X - X[:,best_cfg_i].min(axis=1)[:,None],0)
        best_cfg_i.append(np.argmin(cost_basis.sum(axis=0)))
    c_labels = np.argmin(X[:,best_cfg_i],axis=1)
    return best_cfg_i, c_labels

