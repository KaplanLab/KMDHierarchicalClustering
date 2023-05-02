import numpy as np
from scipy import cluster
import random


class dend_search_tree:
    """
    tree database for hierarchical clustering result
    """
    def __init__(self,tree,parent_node = None):
        self.count = tree.count
        if  self.is_leaf():
            self.right = None
            self.left = None
            self.id = tree.id
            self.dist = tree.dist
            self.parent = parent_node
        else:
            self.right = tree.right
            self.left = tree.left
            self.id = tree.id
            self.dist = tree.dist
            self.parent = parent_node
    def is_leaf(self):
        if self.count == 1:
            return True
        else:
            return False

def cluster_bfs(num_of_clusters,Z_tree,min_size_of_clust):
    """
    broad search in the clustering tree for the first n(num_of_clusters-1) splits that are larger
    then the min_size_of_clust, we assume that a split of two nodes larger then the cluster
    are the root of the clusters sub tree, we save all the sub trees and eliminate the ancestors on the
    next step
    :param num_of_clusters: number of groups
    :param Z_tree: HAC dendrogram in a tree data frame
    :param min_size_of_clust: user difined minimum num of points in a cluster
    :return: cluster roots all the nodes that are splited to clusters
    """
    Z_tree = dend_search_tree(Z_tree)
    cluster_roots = []
    visited = [Z_tree]
    current = Z_tree
    split_count = 0
    split_roots = []
    while(split_count<num_of_clusters-1):
        if current.count < min_size_of_clust :#if the node is a outlier do nothing
            visited = visited
        elif current.is_leaf(): #if the node is a leaf do nothing
            visited = visited
        elif current.left.count>= min_size_of_clust and current.right.count>= min_size_of_clust:# if two splited clusters
            split_count +=1                                                                 # are larger then min size
            cluster_roots += [dend_search_tree(current.left,current),dend_search_tree(current.right,current)] # they are cluster subtrees
            visited = visited + [dend_search_tree(current.left,current),dend_search_tree(current.right,current)] # we save the nodes,and count a split
            split_roots.append(current)
        else:
            left_child = current.left
            right_child =current.right
            visited = visited + [dend_search_tree(left_child,current),dend_search_tree(right_child,current)]


        visited.pop(0)
        if not visited:
            break
        current = visited[0]
    return cluster_roots


def get_clust_dict(list_of_nodes,n):
    """
    get all nodes of cluster subtrees, find there leafs and label them
    :param list_of_nodes: all cluster subtrees
    :return: dict: keys are leafs values are cluster labels
    """
    y_pred = -np.ones(n)
    dict = {}
    clust_num = 1
    list_of_cluster_id = [node.id for node in list_of_nodes]
    cluster_avg_dists_list = []
    cluster_avg_merge_list = []
    for node in list_of_nodes:
        cluster_count = 0
        cluster_dists = 0
        cluster_merge_dist = 0
        visited = [node]
        while len(visited)>0:
            current = visited[0]
            if current.is_leaf():
                y_pred[current.id] = clust_num
            else:
                left_child = dend_search_tree(current.left, current)
                right_child = dend_search_tree(current.right, current)
                if left_child.id in list_of_cluster_id or right_child.id in list_of_cluster_id :# check that node is not ancestor
                    y_pred[np.where(y_pred == clust_num)[0]] = -1
                    clust_num -=1
                    break
                else:
                    visited = visited + [left_child, right_child]
                    cluster_merge_dist += current.parent.dist - current.dist
                    cluster_dists += current.dist
                    cluster_count += 1
            visited.pop(0)
        clust_num +=1
        if cluster_count !=0 :
            cluster_avg_dists_list.append(cluster_dists/cluster_count)
            cluster_avg_merge_list.append(cluster_merge_dist/cluster_count)


    return y_pred, np.mean(cluster_avg_dists_list), np.mean(cluster_avg_merge_list)



def predict_from_Z(Z, n_clusters, min_clust_size, dists, k, certainty=0.5):
    n = dists.shape[0]
    Z_tree = cluster.hierarchy.to_tree(Z)
    list_of_nodes = cluster_bfs(n_clusters,Z_tree,min_clust_size)
    y_pred,all_dists_sum,merge_dists_sum = get_clust_dict(list_of_nodes,n)
    list_of_clusters = [[] for i in range(n_clusters+1)]


    #create list of index for each cluster
    for i in range(y_pred.shape[0]):
        if y_pred[i] == -1:
            list_of_clusters[0].append(i)
        else:
            list_of_clusters[int(y_pred[i])].append(i)


    # predict  cluster of outliers
    outlier_score = []
    outlier_list = []
    idx_list = list(range(len(y_pred)))
    random.shuffle(idx_list)
    if len(list_of_clusters[1])>0:
        for i in idx_list :
            if y_pred[i] == -1 :
                outlier_list.append(i)
                y_pred[i],score = predict_outlier_label(i,dists,list_of_clusters,k)
                outlier_score.append((i,score))
                if score < certainty:
                    y_pred[i] = -1
                else:
                    list_of_clusters[int(y_pred[i])].append(i)

    if not any(list_of_clusters[1:]):
        print("all labels are outliers")
        sil_score = -2
    else:
        lengths = [ x for x in list_of_clusters[1:] if len(x) > 0 ]
        sil_score = kmd_silhouette_score(dists,lengths,k)


    for i in range(len(y_pred)):
        if y_pred[i] >-1:
            y_pred[i]-=1

    return np.array(y_pred,dtype=int),list_of_nodes,all_dists_sum,merge_dists_sum,sil_score, outlier_list

def predict_label(KMDHAC):
    """
    predict clusters and outliers using clustering dendrogeam
    all clusters smaller then min_clust_size are considered as outliers
    :param Z: hac clustering dendrogram(scipy)
    :param n_clusters: number of clusters
    :param min_clust_size: the minimum points in a clusterc
    :param n: nimber of points
    :return: y_pred:cluster assigen
    """
    Z = KMDHAC.Z
    n_clusters = KMDHAC.n_clusters
    min_clust_size = KMDHAC.min_cluster_size
    dists = KMDHAC.dists
    certainty = KMDHAC.certainty
    k = KMDHAC.k
    return predict_from_Z(Z, n_clusters, min_clust_size, dists, k, certainty=certainty)

def predict_outlier_label(outlier_index,dists,list_of_clusters,k):
    """
    predict outliers cluster assigment by minimum distance(k closest neighbor avg dist) from ,main clusters
    :param outlier_index : index of outlier that we want to cluster
    :param dists: distance array
    :param list_of_clusters:list of lists, every list contains a cluster objects indices
    :param k:linkage parameter
    :return: cluster assigment , certainty score
    """
    max_id = 0
    min_dist_from_cluster = np.inf
    all_dists = []
    for cluster_id in range(1,len(list_of_clusters)):
        dist_from_cluster = np.mean(sorted(dists[outlier_index,list_of_clusters[cluster_id]])[0:k])
        if dist_from_cluster<min_dist_from_cluster:
            min_dist_from_cluster = dist_from_cluster
            max_id = cluster_id
        all_dists.append(dist_from_cluster)
    sum_all_dists = sum(sorted(all_dists)[:2])
    return max_id , 1-(min_dist_from_cluster/sum_all_dists)

def kmd_silhouette_score(dists,list_of_clusters,k):
    """
    generalized silhouette-type function to predict clustering performance
    :param dists: distance array
    :param list_of_clusters: list of lists, every list contains a cluster objects indices
    :param k:linkage parameter
    :return: kmd silhouette score
    """
    sil_score_list_avg = []
    sil_score_size = []
    for i in range(len(list_of_clusters)):
        dist_diff = np.zeros((len(list_of_clusters[i]),len(list_of_clusters)-1))
        cluster_array = dists[list_of_clusters[i],:][:,list_of_clusters[i]]
        np.fill_diagonal(cluster_array,np.inf) # we do not want to calculate the distance of leaf to itself
        sorted_dists = np.sort(cluster_array,axis=1)[:,:-1][:,:k]# delete last row of infs
        if sorted_dists.shape[0] == 0:
            continue
        if len(list_of_clusters[i]) == 1:
            a = 0 # distance of cluster to itself of size 1
        else:
            a =np.average(sorted_dists[:,0:k],axis= 1)# sort distance from all cluster leafs, avg only k min dists
        idx = 0
        for j in range(len(list_of_clusters)):
            if len(list_of_clusters[j]) == 0:
                continue
            if i!= j:
                intra_cluster_array = dists[list_of_clusters[i],:][:,list_of_clusters[j]]
                sorted_dists = np.sort(intra_cluster_array, axis=1)[:,:k]
                b = np.average(sorted_dists[:,0:k],axis=1)
                dist_diff[:,idx] = (b-a)
                idx +=1

        sil_score_list_avg.append(np.mean(np.min(dist_diff,axis=1)))
        sil_score_size.append(dist_diff.shape[0])

    return min(sil_score_list_avg)








