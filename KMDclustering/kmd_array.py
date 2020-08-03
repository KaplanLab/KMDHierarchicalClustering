import numpy as np
from statistics import mean

def make_kmd_array(dists, n):
    """
    Initialize array of lists, every entry of the distance array is a list with one value.
    :param dists: distance array
    :param n: num of objects
    :return: nd array of lists , each entry containing initial pair dist
    """
    k_min_dists = np.empty((n,n),dtype=np.object)
    for i in range(n):
        for j in range(n):
            k_min_dists[i][j] = list([dists[i,j]])
    return k_min_dists

def k_min_sparse_topkl(dists, n):
    """
    create array of lists, every entry of the distance array is a list.
    :param dists: distance array
    :param n: num of points
    :param k: max list size
    :return: nd array of lists , each entry containing initial pair dist
    """
    k_min_dists = np.empty((n,n),dtype=np.object)
    for i in range(n):
        for j in range(n):
            k_min_dists[i][j] = list([dists[i,j]])
    return k_min_dists




def merge_clusters(k_dists,x,y,k):

    n = k_dists.shape[0]


    k_dists[:,y] = k_dists[:,y] + k_dists[:,x]
    for i in range(n):
           k_dists[i,y] = sorted(k_dists[i,y])[:k]
    k_dists[y,:] = k_dists[:,y]
    merged_vec = np.array([get_mean_val(k_neigbors_list) for k_neigbors_list in k_dists[:,y]])
    # delete
    k_dists[x,:] = [list([])]*n
    k_dists[:,x] = [list([])]*n
    return k_dists, merged_vec

def get_mean_val(n_list):
    if len(n_list)!=0:
       return mean(n_list)
    else:
        return None