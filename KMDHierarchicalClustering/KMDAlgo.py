import numpy as np
from math import sqrt
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform,cdist
from multiprocessing import Pool
import sys
from sklearn.model_selection import train_test_split
import os
from .kmd_array import merge_clusters
from .predict_clust_label import predict_label
from .cluster_scoring import hungarian_acc
import warnings
import platform
from scipy.spatial.distance import euclidean, correlation

#ignore by message
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")


def nan_euclidean(a,b):
    a = np.array(a,dtype=np.float64)
    b = np.array(b,dtype=np.float64)
    nan_row_idx = np.any(np.vstack((np.isnan(a),np.isnan(b))), axis=0)
    return euclidean(a[~nan_row_idx],b[~nan_row_idx])/a[~nan_row_idx].shape[0]

def nan_correlation(a,b):
    a = np.array(a,dtype=np.float64)
    b = np.array(b,dtype=np.float64)
    nan_row_idx = np.any(np.vstack((np.isnan(a),np.isnan(b))), axis=0)
    if correlation(a[~nan_row_idx],b[~nan_row_idx])/a[~nan_row_idx].shape[0] <0 :
        print(a[~nan_row_idx])
        print(b[~nan_row_idx])
    return correlation(a[~nan_row_idx],b[~nan_row_idx],centered=False)/a[~nan_row_idx].shape[0]


class LinkageUnionFind:
    """Structure for fast cluster labeling in unsorted dendrogram."""
    def __init__(self, n):
        self.parent = np.arange(2 * n - 1, dtype=np.intc)
        self.next_label = n
        self.size = np.ones(2 * n - 1, dtype=np.intc)

    def merge(self, x, y):
        x = int(x)
        y = int(y)
        self.parent[x] = self.next_label
        self.parent[y] = self.next_label
        size = self.size[x] + self.size[y]
        self.size[self.next_label] = size
        self.next_label += 1
        return size

    def find(self, x):
        x = int(x)
        p = x

        while self.parent[x] != x:
            x = self.parent[x]

        while self.parent[p] != x:
            p, self.parent[p] = self.parent[p], x

        return x


class Heap:
    """Binary heap.
    Heap stores values and keys. Values are passed explicitly, whereas keys
    are assigned implicitly to natural numbers (from 0 to n - 1).
    The supported operations (all have O(log n) time complexity):
        * Return the current minimum value and the corresponding key.
        * Remove the current minimum value.
        * Change the value of the given key. Note that the key must be still
          in the heap.
    The heap is stored as an array, where children of parent i have indices
    2 * i + 1 and 2 * i + 2. All public methods are based on  `sift_down` and
    `sift_up` methods, which restore the heap property by moving an element
    down or up in the heap.
    """

    def __init__(self, values):
        self.size = values.shape[0]
        self.index_by_key = np.arange(self.size)
        self.key_by_index = np.arange(self.size)
        self.values = values.copy()


        # Create the heap in a linear time. The algorithm sequentially sifts
        # down items starting from lower levels.
        for i in reversed(range(int(self.size / 2))):
            self.sift_down(i)

    def get_min(self):
        return self.key_by_index[0], self.values[0]

    def remove_min(self):
        self.swap(0, self.size - 1)
        self.size -= 1
        self.sift_down(0)

    def change_value(self, key,value):
       index = self.index_by_key[key]
       old_value = self.values[index]
       self.values[index] = value
       if value < old_value:
            self.sift_up(index)
       else:
            self.sift_down(index)

    def sift_up(self,  index):
        parent = Heap.parent(index)
        while index > 0 and self.values[parent] > self.values[index]:
            self.swap(index, parent)
            index = parent
            parent = Heap.parent(index)

    def sift_down(self,  index):
        child = Heap.left_child(index)
        while child < self.size:
            if (child + 1 < self.size and
                    self.values[child + 1] < self.values[child]):
                child += 1

            if self.values[index] > self.values[child]:
                self.swap(index, child)
                index = child
                child = Heap.left_child(index)
            else:
                break
    @staticmethod
    def left_child(parent):
        return (parent << 1) + 1

    @staticmethod
    def parent(child):
        return (child - 1) >> 1

    def swap(self, i, j):
        self.values[i], self.values[j] = self.values[j], self.values[i]
        key_i = self.key_by_index[i]
        key_j = self.key_by_index[j]
        self.key_by_index[i] = key_j
        self.key_by_index[j] = key_i
        self.index_by_key[key_i] = j
        self.index_by_key[key_j] = i
        
def create_list_of_clusters(y_pred, indexes):
    # create list of indexes for each cluster
    list_of_clusters = [[] for i in range(max(y_pred) + 1)]

    for i, index in enumerate(indexes):
        list_of_clusters[int(y_pred[i])].append(index)
    return list_of_clusters

class KMDClustering:
    def __init__(self, k='compute', n_clusters = 2, min_cluster_size = 'compute', affinity = 'euclidean', certainty = 0.5 ,k_scan_range = range(1,100,3)):
        """
        :param k-number of minimum distances to calculate distance between clusters. if flag is compute, best k will be predicted.
        :param n_clusters - number of clusters
        :param min_cluster_size - the minimum points that can be in a cluster,if cluster is smaller then this size it is
        considered as an outlier
        :param affinity - Metric used to compute the distance. Can be “euclidean”, “correlation”, "spearman",“precomputed",
        or any metric used by scipy.spatial.distance.pdist.If “precomputed”,a distance matrix (instead of a similarity matrix) is needed as input for the fit method
        :param certainty- parameter indicating how certain the algorithm is in the correctness of its classification in
        the outlier hanging step, if 0.5 - all outliers will be hanged if 1 - outliers wikk not be hanged
        :param k_scan_range-iterable of k values to test
        :param y_true-cluster True labels
        :param plot_scores- if True, a plot of intrinsic score vs extrinsic score on different k's will be ploted, True labels
        :param path - path to self prediction for each k , if False - prediction will not be saved
        will be required
        """
        self.certainty = certainty
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.min_cluster_size = min_cluster_size
        self.k = k
        self.k_scan_range = list(k_scan_range)
        self.y_true = []
        self.plot_scores = False
        self.path = False

    def is_nan_values(self,data):
        self.nan_idx = np.all(np.isnan(data),axis = 1 )
        if np.any(np.isnan(data)):
            if type(self.affinity) == str :
                if self.affinity == 'nan_euclidean' or self.affinity == 'nan_correlation':
                    data = data[~self.nan_idx,:]
                else:
                    raise ValueError('input array contains nan values please use suitable method such as nan_euclidean or nan_correlation ')
                    raise SystemExit
        return data

    def calc_dists(self,data, method):
        """
        calaculate distance matrix
        :param data: dataset
        :param method: can be 'spearman', ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.
        :return: distance matrix
        """

        if method == 'precompted':
            return data
        elif method == 'spearman':
            corr_matrix, p_matrix = spearmanr(data, axis=1)
            return np.ones(corr_matrix.shape) - corr_matrix
        try:
            if method == 'nan_euclidean':
                data = squareform(pdist(data, nan_euclidean))
            elif method == 'nan_correlation':
                data = squareform(pdist(data, nan_correlation))
            else:
                data = squareform(pdist(data, method))

        except MemoryError:
            print('MemoryError occurred while calculating a distance matrix')
            print ('Using the subsampling option is recommended')
        return data
    
    def sample_data(self,data,percent_size,seed):
         self.X = data
         x_to_assign,x_sampled= train_test_split(list(range(np.shape(data)[0])),test_size=percent_size,random_state=seed)
         self.dataset = data[x_sampled,:]
         if self.affinity == 'precompted':
             self.dataset = data[:,x_sampled]
         print ('dataset shape after subsampeling' + str(self.dataset.shape))
         self.idx_sampled = x_sampled
         self.idx_to_assign = x_to_assign

    def assign_points(self,y_pred_sub, batch=5000):
        list_of_clusters = create_list_of_clusters(y_pred_sub, self.idx_sampled)
        y_pred = np.empty(self.X.shape[0], dtype=int)
        point_idx_list = self.idx_to_assign
        for point_idx in range(0, np.size(point_idx_list), batch):
            if np.size(point_idx_list) - point_idx < batch:
                batch = np.size(point_idx_list) - point_idx
            min_dist = [np.inf] * batch
            point_id = [-1] * batch

            for cluster_id in range(len(list_of_clusters)):
                if self.X[list_of_clusters[cluster_id]].size == 0:
                    continue
                if self.affinity == 'precomputed':
                    D = self.X[point_idx:point_idx + batch,list_of_clusters[cluster_id]]
                else:
                    D = cdist(self.X[point_idx_list[point_idx:point_idx + batch], :],
                                           self.X[list_of_clusters[cluster_id], :], metric=self.affinity)
                dist_from_cluster_array = np.mean(np.sort(D, axis=1)[:, 0:self.k], axis=1)
                for i in range(batch):
                    if dist_from_cluster_array[i] < min_dist[i]:
                        min_dist[i] = dist_from_cluster_array[i]
                        point_id[i] = cluster_id
                y_pred[point_idx_list[point_idx:point_idx + batch]] = point_id
        for i, val in zip(self.idx_sampled, y_pred_sub):
            y_pred[i] = val
        return y_pred

    def memory_check(self,free_memory):
        memory_model = np.poly1d([2.666e-05,  0.01327 , 75.76]) # fitted memory usage vs data size
        maximum_size = int((memory_model - free_memory).roots[1] )# find positive root of polynom to determine maximum size
        if memory_model(self.n) > free_memory:
            raise MemoryError('Dataset with ' + str(self.n) + ' objects is too large for ' + str(
                free_memory) + 'MB free memory, subsampling to size smaller then ' + str(
                maximum_size) + ' objects is recommended, please specify subsampeling = True, percent_size = # <' + str(
                maximum_size))
            raise SystemExit
  

    def predict_k(self, k_scan_range, y_true=[], plot_scores=False, path=False, runparallel = True, keep_Z_list=False):
        """
        predicting the best k for clustering analysis using the normalized kmd silhuete score
        we run on all k's and find the highest clustering score
        if plot scores is true we plot k vs accuracy score and kmd silhuete score
        :param k_scan_range: iterable of k values
        :param y_true: ground truth clustering labels
        :param plot_scores: can be true or false
        :param path: path to save prediction for each k
        :return: best k for clustering analysis
        """
        min_cluster_size = self.min_cluster_size
        dists = self.dists
        num_of_clusters = self.n_clusters
        n = dists.shape[0]
        in_score_list = []
        ex_score_list = []
        successful_k = []
        Z_list = []
        k_list = np.array(list(k_scan_range))

        for k in k_list:
            print('calculating k='+str(k))
            Z = fast_linkage(dists, n, k)
            Z_list.append(Z)
        if keep_Z_list:
            self.Z_list = Z_list
        for Z,k in zip(Z_list,k_list):
            self.Z = Z 
            self.k = k
            clust_assign, node_list, all_dists_avg, merge_dists_avg, sil_score, outlier_list = predict_label(self)
            if sil_score > -1 :
                in_score_list.append(sil_score)
                successful_k.append(k)
                if plot_scores:
                    ex_score_list.append(hungarian_acc(y_true, clust_assign)[0])

            if path:
                np.save(str(path) + '_k_' + str(k), clust_assign)


        in_score_list = np.array(in_score_list)
        in_score_list = (in_score_list - in_score_list.min()) / (in_score_list.max() - in_score_list.min())
        for i in range(len(successful_k)):
            in_score_list[i] = sqrt(in_score_list[i]) - ((successful_k[i] / n))
        self.sil_score = max(in_score_list)


        if plot_scores:
            plt.figure()
            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_xlabel('k')
            ax1.set_ylabel('in_score', color=color)
            ax1.plot(successful_k, in_score_list, 'o', label='normalized silh score')
            plt.legend()
            ax2 = ax1.twinx()

            color = 'tab:red'
            ax2.set_ylabel('ex_score', color=color)
            ax2.plot(successful_k,ex_score_list, color=color)
            fig.tight_layout()
            plt.savefig('in_and_ex_score_vs_k')
            plt.show()
        best_k_idx = np.argmax(in_score_list)
        self.Z = Z_list[best_k_idx]
        return successful_k[best_k_idx]


    def fit(self,X,sub_sample=False,percent_size=0.2,seed = 1, keep_Z_list=False):
        """
        predict cluster labels using kmd Linkage
        :return:
        clust_assign - cluster for each object
        Z - computed linkage matrix
        outlier list - list of objects classified as outliers
        """
        self.sub_sample = sub_sample
        if sub_sample:
            self.sample_data(X,percent_size,seed)
            self.dataset = self.is_nan_values(self.dataset)
        else:
            self.n = np.shape(X)[0]
            if platform.system() == 'Linux':
                total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
                self.memory_check(free_memory)
            else:
                try:
                    import psutil
                    free_memory= psutil.virtual_memory()
                    free_memory = free_memory.available >> 20
                    self.memory_check(free_memory)

                except ImportError:
                    print ('Warning: It was not possible to perform free memory monitoring,'
                           ' it is recommended to use the subsampling option in medium and large datasets')
                    pass

            X = self.is_nan_values(X)
            self.dataset = X
            self.calc_dists(self.dataset,self.affinity)


        if self.affinity == 'precompted':
            self.dists = self.dataset 
        else:
            self.dists = self.calc_dists(self.dataset,self.affinity)
        self.n = np.shape(self.dists)[0]

        if self.min_cluster_size == 'compute':
            self.min_cluster_size = max(int(self.dataset.shape[0] / (self.n_clusters * 10)), 2)
            print ('Default minimum cluster size is : ' + str(
                self.min_cluster_size) + ' calculated by: max(2,#objects /(10*#clusters)) ')
            print (
                'In general, minimum cluster size can be chosen to be slightly smaller than the size of the smallest expected cluster')

        if self.k == 'compute':
            self.k = self.predict_k(k_scan_range=self.k_scan_range, y_true = self.y_true,plot_scores = self.plot_scores, path= self.path, keep_Z_list=keep_Z_list)
            print ('Predicted k is : '+str(self.k))
        else:
            self.Z = fast_linkage(self.dists, self.n, self.k)

        return self

    def predict(self,X):
        y_pred, node_list, all_dists_avg, merge_dists_avg, sil_score,outlier_list = predict_label(self)
        self.outlier_list = outlier_list
        self.y_pred_sub = y_pred
        self.sil_score = sil_score
        clust_assign = np.zeros(self.nan_idx.size,dtype=int)

        if self.sub_sample: # assign all unclustered objects
            clust_assign = self.assign_points(self.y_pred_sub, batch=5000)
        else:
            clust_assign[self.nan_idx] = -2
            clust_assign[~self.nan_idx] = y_pred
        self.y_pred = clust_assign
        return clust_assign
        

        return clust_assign


def label(Z,  n):
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





# ***************************************************************************************************
# * Function name : find_min_dist
# * Discription   : finds the closest cluster from the rest of the list of clusters
# * Parameters    : n - number of clusters  x - given cluster size - list of cluster sizes D - distance matrix
# * Return value  : y - closest cluster current_min - distance between clusters
# ***************************************************************************************************
def find_min_dist(n , D, size, x):
    current_min = np.inf
    y = -1
    for i in range(x + 1, n):
        if size[i] == 0:
            continue

        dist = D[x,i]
        if dist < current_min:
            current_min = dist
            y = i

    return y, current_min


# ***************************************************************************************************
# * Function name : fast_linkage
# * Discription   : hierarchy clustering using fast linkage algo, at each iteretion min_dist_heap will pop the minimum distance neighbors,
#                   leafs will be clusterd,diatance mat will be updated by the average of K closest neighbors to merged clusters,
#                   neigbors of new cluster will be reasigned as the neigbors of old leafs
# * Parameters    : D - distance mat n - number of leafs K - num of minimum averaged neighbors
#
# * Return value  : Z Computed linkage matrix
# ***************************************************************************************************
def fast_linkage(D,n,K,data =np.array([])):
    Z = np.empty((n - 1, 4))
    size = np.ones(n)  # sizes of clusters

    # generating empty 3D array of the K minimum dists for each new cluster
    K_min_dists =np.empty((n,n),dtype=np.object_)
    dists = D.copy() # Distances between clusters.

    # ID of a cluster to put into linkage matrix.
    cluster_id = np.arange(n,dtype= int)
    neighbor = np.empty(n - 1,dtype= int)
    min_dist = np.empty(n - 1,dtype= np.float64)

    # initializing the heap finding closest neighbor to leaf from the rest of the list of leafs
    for x in range(n - 1):
        neighbor[x], min_dist[x] = find_min_dist(n, dists, size, x)
    min_dist_heap = Heap(min_dist)

    for k in range(n - 1):
        # Theoretically speaking, this can be implemented as "while True", but
        # having a fixed size loop when floating point computations involved
        # looks more reliable. The idea that we should find the two closest
        # clusters in no more that n - k (1 for the last iteration) distance
        # updates.
        for i in range(n - k):
            x, dist = min_dist_heap.get_min()
            y = neighbor[x]

            if dist == dists[x,y]:
                break

            y, dist = find_min_dist(n, dists, size, x)
            neighbor[x] = y
            min_dist[x] = dist
            min_dist_heap.change_value(x, dist)
        min_dist_heap.remove_min()

        id_x = cluster_id[x]
        id_y = cluster_id[y]
        nx = size[x]
        ny = size[y]

        if id_x > id_y:
            id_x, id_y = id_y, id_x

        Z[k, 0] = id_x
        Z[k, 1] = id_y
        Z[k, 2] = dist
        Z[k, 3] = nx + ny

        # update k_min_dists
        K_min_dists,new_cluster_vec = merge_clusters(K_min_dists,dists, x, y, K ,size)
        dists[:, y] = new_cluster_vec
        dists[y, :] = new_cluster_vec


        size[x] = 0  # Cluster x will be dropped.
        size[y] = nx + ny  # Cluster y will be replaced with the new cluster.
        cluster_id[y] = n + k  # Update ID of y.


        # Reassign neighbor candidates from x to y.
        # This reassignment is just a (logical) guess.
        for z in range(x):
            if size[z] > 0 and neighbor[z] == x:
                neighbor[z] = y

        # Update lower bounds of distance.
        for z in range(y):
            if size[z] == 0:
                continue

            dist = dists[z,y]
            if dist < min_dist[z]:
                neighbor[z] = y
                min_dist[z] = dist
                min_dist_heap.change_value(z, dist)

        # Find nearest neighbor for y.
        if y < n - 1:
            z, dist = find_min_dist(n, dists, size, y)
            if z != -1:
                neighbor[y] = z
                min_dist[y] = dist
                min_dist_heap.change_value(y, dist)
    return Z














