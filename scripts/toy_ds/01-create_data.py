from sklearn import cluster, datasets, mixture
import numpy as np
import pandas as pd
import sys

def noisy(n_samples):
    np.random.seed(1)
    # nested circle data
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.3,
                                          noise=0.14)
    # moons dataset
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.24)

    # Anisotropicly distributed data
    random_state = 185
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[2, 2, 2],random_state=random_state)

    return noisy_circles, noisy_moons, aniso, varied

def clean(n_samples):
    np.random.seed(1)
    # nested circle data
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.3,
                                          noise=0.05)
    # moons dataset
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1, 2.5, 0.5], random_state=random_state)

    return noisy_circles, noisy_moons, aniso, varied

if __name__ == '__main__':
    names = ['circles', 'moons', 'bars', 'spheres']
    if 'clean' in sys.argv:
        datasets = clean(1000)
        for name, ds in zip(names, datasets):
            X, y = ds
            df = pd.DataFrame(X, columns=['x1', 'x2'])
            df['label'] = y
            df.to_csv(f'clean_{name}.csv', index_label=False)
    else:
        datasets = noisy(1000)
        for name, ds in zip(names, datasets):
            X, y = ds
            df = pd.DataFrame(X, columns=['x1', 'x2'])
            df['label'] = y
            df.to_csv(f'noisy_{name}.csv', index_label=False)
