import matplotlib.pyplot as plt
import numpy as np

# Tight image, no whitespace around
fig = plt.figure()
fig.set_size_inches(5.28, 3.91)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
# From figure 7
scores = np.array( # Shape: algorithm, scores (triad per dataset)
((0.577, 0.541, 0.488,),
(0.568, 0.529, 0.456,),
(0.512, 0.562, 0.312,),
(0.648, 0.581, 0.468,),)
)

scores = scores.reshape(scores.shape[0], -1, 3)
scores = scores.transpose(1, 0, 2) # shape: dataset, algorithm, score

ds = 0
n_algortihms = scores.shape[1]
algorithms_range = np.arange(n_algorithms)
score_colors = ('#000080', '#FFDAB9', '#008B8B')
scores_range = np.arange(scores.shape[2])
width = 1/(scores.shape[2]+1)
for s in scores_range:
    sc = scores[ds, :, s]
    ax.bar(algorithms_range + width*s, sc, width=width, color=[score_colors[s]] * len(sc))
    plt.xlim(-0.2, n_algoritgms -1 + .7)
    plt.ylim(0, 1.0)
    plt.xticks([])
    plt.yticks([])
plt.savefig(f'sampling_ds{ds}.png')
