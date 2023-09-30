import KMDHierarchicalClustering.predict_clust_label
from KMDHierarchicalClustering.KMDAlgo import KMDClustering
from KMDHierarchicalClustering import cluster_scoring
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import scipy.stats as stats
import warnings
import time 

def normalize_sil(sil, k_range, n):
    norm_sil = (sil - sil.min()) / sil.ptp()
    norm_sil = np.sqrt(norm_sil) - k_range/n
    
    return norm_sil

def clean(n_samples, random_state=5):
    # nested circle data
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.3,
                                          noise=0.05)
    # moons dataset
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05, random_state=random_state)

    # Anisotropicly distributed data
    random_state = 185
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state, cluster_std=0.5)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[0.5, 0.5, 0.5],random_state=random_state)

    return noisy_circles, noisy_moons, aniso, varied

def noisy(n_samples, random_state=5):
    # nested circle data
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.3,
                                          noise=0.14)
    # moons dataset
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.24, random_state=random_state)

    # Anisotropicly distributed data
    random_state = 185
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[2, 2, 2],random_state=random_state)

    return noisy_circles, noisy_moons, aniso, varied

def calc_scores(ds):
    # basic cluster parameters
    default_base = {
                    'n_clusters': 3,
                    'min_cluster_size': 'compute',
                    'k_scan_range':(10,20,5),
                    }
    X, y = ds[0]
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)
    params = default_base.copy()
    params.update(ds[1])
    k_scan_range = params['k_scan_range']
    ret = dict(Z=[], acc=[], nmi=[], ari=[], kmdsil=[], k_range=k_scan_range, mcs=params['min_cluster_size'])
    for k in k_scan_range:
        print(f"Calculating for {k=}:", end=' ')
        kmd = KMDClustering(k=k, n_clusters = params['n_clusters'], min_cluster_size = params['min_cluster_size'], certainty = 0.5)
        kmd.fit(X)
        y_pred, _, _, _, kmdsil, _ = KMDHierarchicalClustering.predict_clust_label.predict_label(kmd)
        acc = cluster_scoring.hungarian_acc(y, y_pred)[0]
        nmi = normalized_mutual_info_score(y, y_pred)
        ari = adjusted_rand_score(y, y_pred)
        ret['Z'].append(kmd.Z)
        ret['kmdsil'].append(kmdsil)
        ret['acc'].append(acc)
        ret['nmi'].append(nmi)
        ret['ari'].append(ari)
        print(f"{acc=} {nmi=} {ari=}")

    ret['kmdsil_norm'] = normalize_sil(np.array(ret['kmdsil']), k_scan_range, X.shape[0])
    ret['acc_corr'] = stats.pearsonr(ret['kmdsil_norm'], ret['acc'])
    ret['nmi_corr'] = stats.pearsonr(ret['kmdsil_norm'], ret['nmi'])
    ret['ari_corr'] = stats.pearsonr(ret['kmdsil_norm'], ret['ari'])
    return ret
    
def main():
    import sys

    FUNCS = dict(clean=clean, noisy=noisy)
    ds_type = sys.argv[1]
    try:
        seed = int(sys.argv[2])
        kwargs = dict(random_state=seed)
        filename = f"sil_scores_wide_moons_{ds_type}_seed{seed}.npz"
        data_filename = f"data_moons_{ds_type}_seed{seed}.npz"
    except:
        kwargs = dict()
        filename = f"sil_scores_wide_moons_{ds_type}.npz"
        data_filename = f"data_moons_{ds_type}.npz"
    ds_func = FUNCS[ds_type]

    print(f"Running {ds_type}, saving scores to {filename} and data to {data_filename}")
    ### generate datasets ###
    n_samples = 1000

    noisy_circles, noisy_moons, aniso, varied = ds_func(n_samples, **kwargs)
    np.savez(data_filename, X=noisy_moons[0], y=noisy_moons[1])

    # note that recomended k_scan_range is 1 to 100, the range was reduced to save time 
    datasets = [
            #(noisy_circles, {'eps': 0.15,'n_clusters': 2, 'k':'compute', 'k_scan_range' : (10,20,5)}),
        (noisy_moons, {'n_clusters': 2, 'k_scan_range': np.arange(1, 101)}),
        #(varied, {'eps': .18, 'n_neighbors': 2,'min_samples': 5, 'min_cluster_size': .2, 'k':'compute', 'k_scan_range' :(10,20,5)}),
        #(aniso, {'eps': .15, 'n_neighbors': 2,'min_samples': 20,  'min_cluster_size': .2, 'k':'compute', 'k_scan_range' : (10,20,5)}),
    ]

    ret = calc_scores(datasets[0])
    np.savez(filename, **ret)

if __name__ == '__main__':
    main()
