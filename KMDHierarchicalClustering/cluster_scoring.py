import numpy as np
from sklearn.metrics import confusion_matrix , accuracy_score
import scipy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import hsv_to_rgb
import warnings
#ignore by message
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", message="Mean of empty slice")


def hungarian_acc(y_true,y_pred):
    """
    Accuracy is calculated by finding an optimal one to one assignment between clusters and true labels
    using the Hungarian algorithm(scipy)
    :param y_true: true assignments
    :param y_pred: predicted assignments
    :return: accuracy score, assignments matching true predictions
    """
    # convert to numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # delete outliers
    core_idx = y_pred != -1
    y_pred_eval = y_pred[core_idx]
    y_eval = y_true[core_idx]

    # calcute confusion matrix, recall, precision and f1
    # some rows/cols of the confusion matrix can be zero, so ignore the warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
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
            new_label=scipy.stats.mode(real_labels[idx], axis=0)[0][0]  # Choose the most common label among data points in the cluster
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



def tsne_presentation(dists,label_list,y_pred):
    y_pred_color_map = []
    y_color_map = []
    y_new_color_map = []
    print (max(max(label_list),max(y_pred)))

    # 'Muted', a colorblind-friendly scheme from https://personal.sron.nl/~pault/#sec:qualitative
    colors = [ '#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499' ]
    color_bad = '#DDDDDD'
    num_colors = len(colors)
    if max(label_list) >= num_colors:
        warnings.warn("More labels than colors. Some colors will be used more than once.")
    _, new_labels = hungarian_acc(label_list, y_pred)
    y_pred= find_matching_labels(label_list,y_pred)
    for clust in label_list:
        if clust == -1 :
            y_new_color_map.append(color_bad)
        else:
            y_color_map.append(colors[clust%num_colors])
    for clust in y_pred:
        if clust == -1 :
            y_pred_color_map.append(color_bad)
        else:
            y_pred_color_map.append(colors[clust%num_colors])
    for clust in new_labels:
        if clust == -1 :
            y_new_color_map.append(color_bad)
        else:
            y_new_color_map.append(colors[clust%num_colors])

    X = dists
    Y = TSNE(n_components=2, perplexity=50, metric='precomputed', init='random').fit_transform(X)

    plt.figure()
    plt.title('Predicted labels')
    plt.scatter(Y[:, 0], Y[:, 1],alpha = 0.3, c=y_new_color_map,edgecolors = 'none')
    plt.show()

    plt.figure()
    plt.title('Matching labels')
    plt.scatter(Y[:, 0], Y[:, 1], c=y_pred_color_map,alpha = 0.3,edgecolors = 'none')
    plt.show()

    plt.figure()
    plt.title('True labels')
    plt.scatter(Y[:, 0], Y[:, 1],alpha = 0.3, c=y_color_map,edgecolors = 'none')
    plt.show()



