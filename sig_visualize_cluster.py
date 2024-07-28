import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved signatures
cam_1_sig = np.loadtxt('sig_enc/cam_1_sig')
cam_2_sig = np.loadtxt('sig_enc/cam_2_sig')
cam_3_sig = np.loadtxt('sig_enc/cam_3_sig')
cam_4_sig = np.loadtxt('sig_enc/cam_4_sig')
cam_5_sig = np.loadtxt('sig_enc/cam_5_sig')
cam_6_sig = np.loadtxt('sig_enc/cam_6_sig')
cam_7_sig = np.loadtxt('sig_enc/cam_7_sig')

# Combine all signatures and create labels
all_sigs = np.vstack([cam_1_sig, cam_2_sig, cam_3_sig, cam_4_sig, cam_5_sig, cam_6_sig, cam_7_sig])
labels = np.array([1] * cam_1_sig.shape[0] + [2] * cam_2_sig.shape[0] + [3] * cam_3_sig.shape[0] + [4] * cam_4_sig.shape[0] + [5] * cam_5_sig.shape[0] + [6] * cam_6_sig.shape[0] + [7] * cam_7_sig.shape[0],)

# PCA Visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_sigs)

plt.figure(figsize=(16, 10))
sns.scatterplot(
    x=pca_result[:, 0], y=pca_result[:, 1],
    hue=labels,
    palette=sns.color_palette(n_colors=8),
    legend="full",
    alpha=0.6
)
plt.title("PCA of Camera Signatures")
plt.show()

# t-SNE Visualization
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
tsne_result = tsne.fit_transform(all_sigs)

plt.figure(figsize=(16, 10))
sns.scatterplot(
    x=tsne_result[:, 0], y=tsne_result[:, 1],
    hue=labels,
    palette=sns.color_palette(n_colors=8),
    legend="full",
    alpha=0.6
)
plt.title("t-SNE of Camera Signatures")
plt.show()
