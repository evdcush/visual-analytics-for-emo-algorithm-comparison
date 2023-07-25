import numpy as np
import hdbscan

def clusterPF(PF, name):
    if name == 'DTLZ7':
        return hdbscan.HDBSCAN(min_cluster_size=100, min_samples=20).fit_predict(PF).tolist()
    else:
        return np.ones(len(PF)).tolist()