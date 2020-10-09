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

def plot_mds_points(mds_y, points, labels=None, style='rx-'):
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


#  Representational dissimilarity matrices, based on a list of vectors.
def rdm_spearman(vectors):
    matrix = np.zeros((len(vectors), len(vectors)))
    for i, vec1 in enumerate(vectors):
        for j, vec2 in enumerate(vectors):
            matrix[i, j] = 1 - stats.spearmanr(vec1, vec2)[0]
    return matrix


def plot_rdm(matrix, labels, title, show_rdm=False):
    sns.heatmap(matrix, cbar=True, square=True, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    if show_rdm:
        plt.show()

def compare_matrices(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        raise ValueError("both matrices must be the same shape")
    return stats.spearmanr(matrix1.flatten(), matrix2.flatten())[0]