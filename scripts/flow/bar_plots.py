import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5.41, 3.84))
# From table S4
scores = np.array( # Shape: algorithm, scores (triad per dataset)
((0.5587, 0.7071, 0.5642, 0.5702, 0.7252, 0.6169, 0.4819, 0.6506, 0.4628),
(0.7824, 0.7849, 0.7722, 0.8706, 0.8646, 0.8958, 0.9100, 0.8702, 0.8890),
(0.6918, 0.6679, 0.6559, 0.8922, 0.8416, 0.9273, 0.8253, 0.7243, 0.8251),
(0.5591, 0.7224, 0.5911, 0.5218, 0.6644, 0.4811, 0.5935, 0.7279, 0.5664),
(0.8436, 0.8443, 0.8579, 0.8646, 0.8725, 0.8734, 0.6331, 0.6599, 0.6188),
(0.9180, 0.8827, 0.9268, 0.6598, 0.7607, 0.6719, 0.9235, 0.8996, 0.9248),
(0.8384, 0.8507, 0.7963, 0.9296, 0.9357, 0.9594, 0.8940, 0.8412, 0.8626),))
stddev = np.array(
((0.0303, 0.0139, 0.0367, 0.0550, 0.0213, 0.0755, 0.0585, 0.0201, 0.0512),
(0.0216, 0.0205, 0.0298, 0.0510, 0.0405, 0.04278, 0.0230, 0.0063, 0.0281),
(0.0141, 0.0094, 0.0107, 0.0023, 0.0030, 0.0031, 0.0067, 0.0059, 0.0079),
(0.1203, 0.061, 0.129, 0.0947, 0.0590, 0.1048, 0.0714, 0.0383, 0.0767),
(0.0272, 0.0143, 0.0209, 0.0903, 0.0596, 0.1195, 0.0666, 0.0460, 0.0833),
(0.0014, 0.0021, 0.0032, 0.0412, 0.0220, 0.0368, 0.0423, 0.0257, 0.0524),
(0.0092, 0.0177, 0.0208, 0.0105, 0.0066, 0.0070, 0.0037, 0.0050, 0.0063),)
)
scores = scores.reshape(scores.shape[0], -1, 3)
scores = scores.transpose(1, 0, 2) # shape: dataset, algorithm, score
stddev = stddev.reshape(stddev.shape[0], -1, 3)
stddev = stddev.transpose(1, 0, 2) # shape: dataset, algorithm, score

ds = 2
algorithms_range = np.arange(scores.shape[1])
score_colors = ('#000080', '#FFDAB9', '#008B8B')
scores_range = np.arange(scores.shape[2])
width = 1/(scores.shape[2]+1)
for s in scores_range:
    sc = scores[ds, :, s]
    err = stddev[ds, :, s]
    plt.bar(algorithms_range + width*s, sc, width=width, color=[score_colors[s]] * len(sc), yerr=err, capsize=3)
    plt.xlim(-0.2, 6.7)
    plt.ylim(0, 1.01) # For first dataset, use ylim=0.91
    plt.gca().axis('off')

plt.show()
