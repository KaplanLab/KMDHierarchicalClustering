import scanpy as sc



def filter_genes(data,precent):
    """
    filter over and under expressed genes
    """
    print (data.shape)
    n = int(data.shape[0]*(precent/100))
    print (n)
    sc.pp.filter_genes(data,max_cells = data.shape[0]-n)
    sc.pp.filter_genes(data,min_cells = n)
    print (data.shape)


def obs_names_to_numbers(keys,labels):
    """
    convert observation names to numbers
    :param keys: cell groups
    :param labels: cell labels
    :return: label list - cell number group of each cell
             organ_idx - index of cell
    """
    obs_dict= dict( zip( keys, list(range(1,len(keys)+1))))
    label_list = []
    organ_idx = []
    for i in range(labels.shape[0]):
        for key in obs_dict:
            if labels[i].find(key)!= -1:
                label_list.append(obs_dict[key])
                organ_idx.append(i)
    if len(label_list)!= labels.shape[0]:
        print ('original label list contains unidentified keys')

    return label_list, organ_idx
