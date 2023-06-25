import matplotlib.pyplot as plt
import numpy as np

# Tight image, no whitespace around
fig = plt.figure()
fig.set_size_inches(5.28, 3.91)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
# From table S7
scores = np.array( # Shape: algorithm, scores (triad per dataset)
((0.674, 0.699, 0.576, 0.623, 0.728, 0.496, 0.729, 0.764, 0.592),
(0.686, 0.723, 0.583, 0.52,  0.683, 0.400, 0.713, 0.783, 0.586),
(0.808, 0.76,  0.769, 0.711, 0.731, 0.586, 0.582, 0.5, 0.341),
(0.657, 0.678, 0.658, 0.611, 0.716, 0.488, 0.715, 0.687, 0.544),
(0.893, 0.790, 0.831, 0.738, 0.686, 0.523, 0.925, 0.882, 0.885),)
)

scores = scores.reshape(scores.shape[0], -1, 3)
scores = scores.transpose(1, 0, 2) # shape: dataset, algorithm, score

ds = 2
algorithms_range = np.arange(scores.shape[1])
score_colors = ('#000080', '#FFDAB9', '#008B8B')
scores_range = np.arange(scores.shape[2])
width = 1/(scores.shape[2]+1)
for s in scores_range:
    sc = scores[ds, :, s]
    ax.bar(algorithms_range + width*s, sc, width=width, color=[score_colors[s]] * len(sc))
    plt.xlim(-0.2, 4.7)
    plt.ylim(0, 1.0)
    plt.xticks([])
    plt.yticks([])
plt.savefig(f'bars_ds{ds}.png')
