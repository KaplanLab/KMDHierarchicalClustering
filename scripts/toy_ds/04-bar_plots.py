import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5, 5.99))
# From table S2
scores = np.array(( # Shape: algorithm, scores (triad per dataset)
(0.711, 0.289, 0.177, 0.853, 0.403, 0.5, 0.899, 0.665, 0.726, 0.91, 0.741, 0.76),
(0.687, 0.237, 0.139, 0.839, 0.41, 0.459, 0.632, 0.514, 0.428, 0.668, 0.722, 0.568),
(0.501, 0.002, 0, 0.501, 0.002, 0, 0.335, 0.004, 0, 0.335, 0.004, 0),
(0.667, 0.536, 0.721, 0.494, 0.352, 0, 0.342, 0.033, 0, 0.67, 0.724, 0.567),
(0, 0, 0, 0.941, 0.678, 0.778, 0.924, 0.754, 0.794, 0.667, 0.736, 0.569),
(0.552, 0.009, 0.011, 0.834, 0.353, 0.457, 0.923, 0.727, 0.784, 0.998, 0.976, 0.988),
(0.990, 0.922, 0.960, 0.915, 0.581, 0.689, 0.914, 0.717, 0.763, 0.971, 0.881, 0.916),
(0.992, 0.932, 0.967, 0.929, 0.631, 0.736, 0.916, 0.726, 0.772, 0.995, 0.973, 0.984),
))
scores = scores.reshape(scores.shape[0], -1, 3)  # shape: algorithm, dataset, scores (acc, nmi, ari)
scores = scores.transpose(1, 0, 2) # shape: dataset, algorithm, score
ds = 3
algorithms_range = np.arange(scores.shape[1])
score_colors = ('#000080', '#FFDAB9', '#008B8B')
scores_range = np.arange(scores.shape[2])
width = 1/(scores.shape[2]+1)
for s in scores_range:
    sc = scores[ds, :, s]
    plt.bar(algorithms_range + width*s, sc, width=width, color=[score_colors[s]] * len(sc))
    plt.xlim(-0.2, 7.7)
    plt.ylim(0, 1.01)
    plt.gca().axis('off')
plt.show()
