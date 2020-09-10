import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.manifold import MDS


import seaborn as sns


#Multidimensional scaling
def mds(vectors):
    # Make array from vectors
    vectors_array = np.stack(vectors, axis=0)

    # Set-up manifold methods
    mds = MDS(n_components=2, metric=True)

    # Plot results
    Y = mds.fit_transform(vectors_array)
    return Y

def plot_mds_points(mds_y, points, labels=None, style='rx-', show=True, title="-"):
    toplotx = [mds_y[point, 0] for point in points]
    toploty = [mds_y[point, 1] for point in points]

    for i in range(len(toplotx)):
        x = toplotx[i]
        y = toploty[i]
        if labels == None:
            label = str(i)
        else:
            label = labels[i]
        plt.text(x * (1 + 0.01), y * (1 + 0.01), label, fontsize=9)

    plt.plot(toplotx, toploty, style)
    if show:
        plt.title(title + "MDS")
        plt.show()

#  Representational dissimilarity matrices, based on a list of vectors.
def rdm_spearman(vectors):
    matrix = np.zeros((len(vectors), len(vectors)))
    for i, vec1 in enumerate(vectors):
        for j, vec2 in enumerate(vectors):
            matrix[i, j] = 1 - stats.spearmanr(vec1, vec2)[0]
    return matrix


def show_rdm(matrix, labels, title):
    sns.heatmap(matrix, cbar=True, square=True, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.show()
