from typing import Callable, Iterable, Optional, Union
import sys

import numpy as np
import pandas as pd
import scipy as sp
import random

from KMDHierarchicalClustering import KMDClustering, predict_clust_label
import DBCV

class ClusterNumberEstimator:
    def __init__(self, k_range: Iterable[int], X: np.ndarray,
                 Z_list: Iterable[np.ndarray], dists: Union[Callable, str],
                 min_cluster_size: int, n_features: Optional[int]=None):
        self.k_range = np.fromiter(k_range, dtype=int)
        self.Z_dict = { k: Z for k, Z in zip(self.k_range, Z_list) }
        if dists == 'precomputed':
            assert n_features is not None
            self.n_features = n_features
            self.dists = X
        else:
            self.dists = sp.spatial.distance.cdist(X, X, metric=dists)
            self.n_features = X.shape[1]
        self.n = self.dists.shape[0]
        self.min_cluster_size = min_cluster_size

        assert len(self.k_range) == len(self.Z_dict)

    def _calc_sils(self, c, seed=1, debug=False):
        silhouettes = []
        successful_k = []
        failed = []
        for k in self.k_range:
            random.seed(seed)
            ret =  predict_clust_label.predict_from_Z(self.Z_dict[k], c, self.min_cluster_size, self.dists, k)
            sil = ret[4]
            if sil > -1:
                silhouettes.append(sil)
                successful_k.append(k)
            else:
                failed.append(k)
            msg = f"{k=} silhouette={silhouettes[-1]:.4f} (failed k: {failed})"
            if debug:
                msg += '\n'
            print(msg, end=f'\033[K\033[{len(msg)}D', file=sys.stderr)
            sys.stderr.flush()
        normalized_sils = predict_clust_label.normalize_kmd_silhouette(silhouettes, successful_k, self.n)

        return normalized_sils

    def _calc_dbcv(self, c, k):
        Z = self.Z_dict[k]
        labels, *_ = predict_clust_label.predict_from_Z(Z, c, self.min_cluster_size, self.dists, k, certainty=1)
        core_mask = labels != -1
        dists_nooutliers = self.dists[core_mask, :][:, core_mask] 
        labels_nooutliers = labels[core_mask]
        
        return DBCV.DBCV(dists_nooutliers, labels_nooutliers, self.n_features)

    def _c_scores(self, c_range: Iterable[int], random_state:int=1, debug=False):
        c_scores = {}
        for c in c_range:
            print(f"\rCalculating {c=}:", end=' ', file=sys.stderr)
            sils = self._calc_sils(c, seed=random_state, debug=debug)
            best = np.argmax(sils)
            k = self.k_range[best]
            dbcv_score = self._calc_dbcv(c, k)
            print(f"\rFinished {c=}: {k=} and dbcv={dbcv_score:.4f}\033[K", file=sys.stderr)
            c_scores[c] = (k, dbcv_score)

        if debug:
            print()
            print("Results:")
            for c, scores in c_scores.items():
                print(f"{c=} -> [k, dbcv]={scores}")

        return c_scores

    def _get_best_score(self, c_scores):
        best_dbcv = -np.inf
        best_c = None
        best_k = None

        best_item = max(c_scores.items(), key=lambda a: a[1][1])
        best_c, (best_k, best_dbcv) = best_item

        return best_c, best_k, best_dbcv

    def run(self, c_range: Iterable[int], random_state:int=1, debug=False):
        """
        Find the number of clusters in the given range with the best DBCV score.

        Returns a tuple of (best_c, best_k, dbcv_score)
        """
        scores = self._c_scores(c_range, debug=debug)
        return self._get_best_score(scores)

if __name__ == '__main__':
    c_range = np.arange(2, 15)

    job_name = 'Li17'
    fit_filename = f'{job_name}_fit.npz'
    data_filename = f'{job_name}_ds.npz'
    output_filename = f'{job_name}_cluster_number_estimation.npz'

    print("Loading fit file", fit_filename)
    fit_file = np.load(fit_filename)
    Z_list = fit_file['Z']
    k_range = fit_file['k_range']
    mcs = fit_file['mcs']
    X = fit_file['dists'].squeeze()
    dists = 'precomputed'
    print("Loading data", data_filename)
    npz = np.load(data_filename)
    n_features = npz['X'].shape[1]

    print("Calculating silhouette and DBCV metrics over cluster numbers:", c_range)
    estimator = ClusterNumberEstimator(k_range, X, Z_list, dists, mcs, n_features)
    c, k, dbcv = estimator.run(c_range, debug=False)

    print(f"Best combination: {c=} {k=} {dbcv=:.4f}")
    print(f"Saving to {output_filename}")
    np.savez(output_filename, c=c, k=k, dbcv=dbcv)
