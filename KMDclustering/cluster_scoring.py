import numpy as np
from sklearn.metrics import confusion_matrix , accuracy_score
import scipy

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





