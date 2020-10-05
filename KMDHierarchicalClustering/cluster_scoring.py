import numpy as np
from scipy.cluster.hierarchy import dendrogram, maxdists
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix , accuracy_score
import warnings
#ignore by message
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")


import scipy
def find_permutation(real_labels, labels):
    """
    this function runs on all cluster predicted labels and finds the label that is the most frequent in the true labels
    :param real_labels: ground truth labels
    :param labels: predicted labels
    :return: the label that is most likely to match the grounf truth labels
    """
    permutation=[]
    for i in range(max(labels)+1):
        idx = labels == i
        if any(idx):
            new_label=scipy.stats.mode(real_labels[idx])[0][0]  # Choose the most common label among data points in the cluster
            permutation.append(new_label)
        else:
            permutation.append(-1)
    return permutation

def find_matching_labels(y,y_pred):
    """
    this function first permutes all predictions to the most likely original cluster
    and calculates accurecy after changing to the new labels
    it will not ditecet a cluster that was split/
    :param y: ground tuth labels
    :param y_pred: predicted labels
    :return: accuracy score
    """
    permutation = find_permutation(np.array(y),np.array(y_pred))
    new_labels = [permutation[label] for label in y_pred]  # permute the labels
    return new_labels


def n_largest_clusters(y_pred,y_true,n_clusters):
    '''
    calculate cluster assigment of clustering algo, by comperimg to known labels
    we first find n largest clusters and then check when is the cluster assignment best compered to labels
    if leaf is not assigned to any cluster it is an outlier and gets a -1 penalty
    :param y_pred: algo clustering assignment
    :param y_true: ground truth clustering assignment
    :param n_clusters: number of clustrs in dataset
    :return:
    '''
    y_pred = np.array(y_pred)
    big_cluster_list = []
    label_assigment = y_pred.copy()
    for i in range(n_clusters):
        max_clust = max(y_pred,key=list(y_pred).count)
        big_cluster_list.append( np.where(label_assigment == max_clust)[0])
        y_pred = y_pred[np.where(y_pred != max_clust)[0]]
        if len(y_pred)==0:
            break

    num_in_clust= []
    for cluster in big_cluster_list:
        row = []
        # cluster = list(cluster[0])
        for i in range(0,n_clusters):
            sum = 0
            group = np.where(y_true == i)[0]
            for num in (cluster):
                if num in (group):
                    sum +=1
            row.append(sum)
        num_in_clust.append(row)

    # find best combination
    r = [[]]
    for x in num_in_clust:
        print ('finding best combination')
        r = [i + [y] for y in x for i in r]

    score = (max(np.sum(np.array(r),axis = 1))/(len(y_true)))

    print ('exscore')
    print(score)
    return (score)

def intrinsic_score(clust_node_list,method = 'edge_sum'):
    edge_sum = 0
    count = 0
    significant_root = clust_node_list[0].parent
    if method == 'mean_path_length':
        for node in reversed(clust_node_list):
            count += 1
            while node != significant_root: # we stop calculating in first "significant root"
                edge_sum += node.parent.dist - node.dist
                if node.parent in clust_node_list:
                    clust_node_list.remove(node.parent)
                node = node.parent

    if method == 'min_edge':
        for i in range(0,len(clust_node_list),2): # we have list of two children of each parent, we choose the min diff
            edge_sum +=clust_node_list[i].parent.dist - max(clust_node_list[i].dist,clust_node_list[i+1].dist)
            count +=1
    if method == 'edge_sum':
        counted_node_list = []
        for node in reversed(clust_node_list):
            count += 1
            while node != significant_root: # we stop calculating in first "significant root"
                edge_sum += node.parent.dist - node.dist
                if node in counted_node_list: # if dists edges where already summed
                    break
                counted_node_list.append(node)
                node = node.parent


    return  edge_sum/count

def calc_tree_cost(Z,dists):
    score = 0
    n = Z.shape[0]+1
    leaf_list = [[] for i in range(n)]
    for i,row in enumerate(Z):
        all_dists = []
        if row[0] < n :
            x_leafs = [row[0]]
        else:
            x_leafs = leaf_list[int(row[0]-n)]
            leaf_list[int(row[0] - n)] = []
        if row[1]<n :
            y_leafs = [row[1]]
        else:
            y_leafs = leaf_list[int(row[1]-n)]
            leaf_list[int(row[1] - n)] = []
        for x_leaf in x_leafs:
            for y_leaf in y_leafs:
                all_dists.append(dists[int(x_leaf),int(y_leaf)])
        if len(all_dists)>0:
            score += np.sum(all_dists)*row[3]
        leaf_list[i] = x_leafs + y_leafs
    return score

def calc_kmd_tree_cost(Z,dists,k):
    score = 0
    n = Z.shape[0]+1
    leaf_list = [[] for i in range(n)]
    for i,row in enumerate(Z):
        all_dists = []
        if row[0] < n :
            x_leafs = [row[0]]
        else:
            x_leafs = leaf_list[int(row[0]-n)]
            leaf_list[int(row[0] - n)] = []
        if row[1]<n :
            y_leafs = [row[1]]
        else:
            y_leafs = leaf_list[int(row[1]-n)]
            leaf_list[int(row[1] - n)] = []
        for x_leaf in x_leafs:
            for y_leaf in y_leafs:
                all_dists.append(dists[int(x_leaf),int(y_leaf)])
        if len(all_dists)>0:
            score += np.sum(np.sort(all_dists)[:k])*row[3]
        leaf_list[i] = x_leafs + y_leafs
    return score

def hungarian_acc(y_true,y_pred):
    # convert to numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # match y_true index to y pred
    if y_true.min() == 1:
        y_true = y_true - np.ones(y_pred.size)
    # delete outliers
    core_idx = y_pred != -1
    y_pred_eval = y_pred[core_idx]
    y_eval = y_true[core_idx]

    # calcute confusion matrix, recall, precision and f1
    conf = confusion_matrix(y_eval,y_pred_eval)
    Recall_array = conf/conf.sum(axis=1, keepdims=True)
    Recall_array[np.isnan(Recall_array)] = 0
    precision_array = conf/conf.sum(axis=0, keepdims=True)
    precision_array[np.isnan(precision_array)] = 0
    F_array =(2*Recall_array*precision_array)/(precision_array+Recall_array)
    F_array[np.isnan(F_array)] = 0
    F_array = np.ones(precision_array.shape)-F_array

    # hungarian optimization
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(F_array)

    # match true labels to predicted labels
    new_label_dict = dict(zip(col_ind,row_ind))

    for i in range(y_pred.shape[0]):
        if y_pred[i] != -1:
            y_pred[i] = new_label_dict[y_pred[i]]

    return accuracy_score(y_true,y_pred),y_pred





