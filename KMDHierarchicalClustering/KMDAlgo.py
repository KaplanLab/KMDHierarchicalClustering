import numpy as np
from math import sqrt
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from multiprocessing import Pool
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

from .kmd_array import make_kmd_array, merge_clusters
from .predict_clust_label import predict
from .cluster_scoring import hungarian_acc
import warnings
#ignore by message
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")


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
    def __init__(self, k='compute', n_clusters = 2, min_cluster_size = 'compute', affinity = 'euclidean', certainty = 0.5 ,k_scan_range = (1,100,3), y_true = [], plot_scores=False,path=False):
        """
        :param k-number of minimum distances to calculate distance between clusters. if flag is compute, best k will be predicted.
        :param n_clusters - number of clusters
        :param min_cluster_size - the minimum points that can be in a cluster,if cluster is smaller then this size it is
        considered as an outlier
        :param affinity - Metric used to compute the distance. Can be “euclidean”, “correlation”, "spearman",“precomputed",
        or any metric used by scipy.spatial.distance.pdist.If “precomputed”,a distance matrix (instead of a similarity matrix) is needed as input for the fit method
        :param certainty- parameter indicating how certain the algorithm is in the correctness of its classification in
        the outlier hanging step, if 0.5 - all outliers will be hanged if 1 - outliers wikk not be hanged
        :param k_scan_range-(tuple) the range of k's used to search for k.(start k, stop k, jumps)
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
        self.k_scan_range = k_scan_range
        self.y_true = []
        self.plot_scores = False
        self.path = False


    def clac_dists(self,data, method):
        """
        calaculate distance matrix
        :param data: dataset
        :param method: can be 'spearman', ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.
        :return: distance matrix
        """
        if method == 'spearman':
            corr_matrix, p_matrix = spearmanr(data, axis=1)
            return np.ones(corr_matrix.shape) - corr_matrix
        return squareform(pdist(data, method))
    
    def sample_data(self,data,percent_size,seed):
         self.X = data
         x_to_assign,x_sampled= train_test_split(list(range(np.shape(data)[0])),test_size=percent_size,random_state=seed)
         self.dataset = data[x_sampled,:]
         print (self.dataset.shape)
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
                D = pairwise_distances(self.X[point_idx_list[point_idx:point_idx + batch], :],
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
  

    def predict_k(self, min_k= 1, max_k = 100, y_true=[], plot_scores=False, path=False, k_jumps=3, runparallel = True):
        """
        predicting the best k for clustering analysis using the normalized kmd silhuete score
        we run on all k's and find the highest clustering score
        if plot scores is true we plot k vs accuracy score and kmd silhuete score
        :param min_k: minimum k
        :param max_k: maximum k
        :param k_jumps: an integer number specifying the incrementation
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
        k_list = list(range(min_k, max_k, k_jumps))
        k_min_dists = make_kmd_array(dists, n)

        Z_list = []
        for k in k_list:
            Z = fast_linkage(dists, n, k)
            Z_list.append(Z)
        for Z in Z_list:
            clust_assign, node_list, all_dists_avg, merge_dists_avg, sil_score, outlier_list = predict(
                Z, num_of_clusters, min_cluster_size, dists, k)
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
        return successful_k[np.argmax(in_score_list)]


    def fit(self,X,sub_sample=False,percent_size=0.2,seed = 1):
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
        else:
            self.dataset = X
        self.calc_dists(self.dataset,self.affinity)

        if self.affinity == 'precompted':
            self.dists = X
        else:
            self.dists = self.clac_dists(X,self.affinity)

        if self.min_cluster_size == 'compute':
            self.min_cluster_size = max(int(self.dataset.shape[0] / (self.n_clusters * 10)), 2)
            print ('Default minimum cluster size is : ' + str(
                self.min_cluster_size) + ' calculated by: max(2,#objects /(10*#clusters)) ')
            print (
                'In general, minimum cluster size can be chosen to be slightly smaller than the size of the smallest expected cluster')

        if self.k == 'compute':
            self.k = self.predict_k( min_k= self.k_scan_range[0], max_k = self.k_scan_range[1],k_jumps= self.k_scan_range[2],y_true = self.y_true,plot_scores = self.plot_scores, path= self.path )
            print ('Predicted k is : '+str(self.k))


        dists =self.dists
        n = np.shape(dists)[0]
        self.Z = fast_linkage(self.dists, n, self.k)

    def predict(self,X):
        clust_assign, node_list, all_dists_avg, merge_dists_avg, sil_score,outlier_list = predict(self.Z, self.n_clusters,self.min_cluster_size, self.dists, self.k, self.certainty )
        self.outlier_list = outlier_list

        return clust_assign

    def predict_M(self,y_true):
        sil_score_list = []
        M_list = []
        accuracy_list = []
        for M in range(2,500,10):
            clust_assign, node_list, all_dists_avg, merge_dists_avg, sil_score,outlier_list = predict(self.Z, self.n_clusters,M, self.dists, self.k, self.certainty )
            sil_score_list.append(sil_score)
            M_list.append(M)
            accuracy_list.append(hungarian_acc(y_true,clust_assign))
        np.save('sil_score_m',sil_score_list)
        np.save('M_list', M_list)
        np.save('accuracy_list',accuracy_list)
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
    # generating 3D array of the K minimum dists for each new cluster
    if data.shape[0] == 0 :
        K_min_dists = make_kmd_array(D, n)
    else:
        K_min_dists = data

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

        size[x] = 0  # Cluster x will be dropped.
        size[y] = nx + ny  # Cluster y will be replaced with the new cluster.
        cluster_id[y] = n + k  # Update ID of y.

        # update k_min_dists
        K_min_dists,new_cluster_vec = merge_clusters(K_min_dists, x, y, K)
        dists[:, y] = new_cluster_vec
        dists[y, :] = new_cluster_vec


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














