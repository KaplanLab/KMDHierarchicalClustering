import numpy as np
from statistics import mean
import array

def make_kmd_array(dists, n):
    """
    Initialize array of lists, every entry of the distance array is a list with one value.
    :param dists: distance array
    :param n: num of objects
    :return: nd array of lists , each entry containing initial pair dist
    """
    print ('creating array')
    k_min_dists = np.empty((n,n),dtype=np.object)
    for i in range(n):
        for j in range(n):
            k_min_dists[i][j] = array.array('f',[dists[i,j]])
    print ('array initialized')
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
            k_min_dists[i][j] = array.array([dists[i,j]])
    return k_min_dists


def merge_clusters(k_dists,dists,x,y,k,size):
    n = k_dists.shape[0]
    for i in range(n):
        if size[i] == 1.0 and size[y] == 1.0:
            y_array = array.array('f',[dists[i,y]])
        else:
            y_array = k_dists[i,y]

        if size[i] == 1.0 and size[x] == 1.0:
            x_array = array.array('f',[dists[i,x]])
        else:
            x_array = k_dists[i,x]

        k_dists[i,y] = merge_arrays(y_array,x_array,k)
    k_dists[y,:] = k_dists[:,y]
    merged_vec = np.array([get_mean_val(k_neigbors_list) for k_neigbors_list in k_dists[:,y]])
    # delete
    k_dists[x,:] = [array.array('f',[])]*n
    k_dists[:,x] = [array.array('f',[])]*n
    return k_dists, merged_vec

def merge_arrays(arr1,arr2,k):
    n = len(arr1)+len(arr2)
    array_size = min(n,k) # maximum number of neigbors needed for each elemnt is k
    arr3 = array.array('f')
    i = 0
    j = 0
    for x in range(array_size):
        if arr1[i] < arr2[j]:
            arr3.append(arr1[i])
            i = i + 1
        else:
            arr3.append(arr2[j])
            j = j + 1
        if i == len(arr1):
            arr3.extend(arr2[j:j+array_size-(x+1)])
            break
        if j == len(arr2):
            arr3.extend(arr1[i:i+array_size-(x+1)])
            break

    return array.array('f',arr3)

def get_mean_val(n_list):
    if len(n_list)!=0:
       return mean(n_list)
    else:
        return None
