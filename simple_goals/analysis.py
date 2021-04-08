import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import utils

import seaborn as sns

SPEARMAN="spearman"
MAHALANOBIS = "mahalanobis"
EUCLIDIAN = "euclidian"
CRAPPYNOBIS = "crappynobis"

#Multidimensional scaling
def mds(vectors):
    # Make array from vectors
    vectors_array = np.stack(vectors, axis=0)

    # Set-up manifold methods
    mds = MDS(n_components=2, metric=True)
    # Plot results
    Y = mds.fit_transform(vectors_array)
    print(mds.stress_)
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
    # Added because very large RDMs take a lot of time to compute
    progress_bar = utils.ProgressBar()
    for i, vec1 in enumerate(vectors):
        if len(vectors)>100:
            progress_bar.updateProgress(i, len(vectors), prefix="Generating RDM:")
        for j, vec2 in enumerate(vectors):
            matrix[i, j] = 1 - stats.spearmanr(vec1, vec2)[0]
    return matrix

def rdm_euclidian(vectors):
    matrix = np.zeros((len(vectors), len(vectors)))
    for i, vec1 in enumerate(vectors):
        for j, vec2 in enumerate(vectors):
            matrix[i, j] = np.sqrt(np.sum((vec1-vec2)**2))
    return matrix

def rdm_mahalanobis(vectors):
    vectors_mat = np.asarray(vectors)
    covmat = np.cov(np.transpose(vectors_mat))
    invcovmat = np.linalg.inv(covmat)
    matrix = np.zeros((len(vectors), len(vectors)))
    for i, vec1 in enumerate(vectors):
        for j, vec2 in enumerate(vectors):
            matrix[i, j] = np.matmul(np.transpose(vec1-vec2), np.matmul(invcovmat, (vec1-vec2)))
    return matrix

def rdm_noisy_mahalanobis(vectors):
    """
    :param vectors: list of lists vectors (unlike other rdm functions which are just lists of vectors)
    """
    # 1. Compute the invcovmat based on everything
    #vectors_mat = np.asarray(vectors)
    #vectors_mat = vectors_mat.reshape(-1, *vectors_mat.shape[2:])
    #covmat = np.cov(np.transpose(vectors_mat))
    #invcovmat = np.linalg.inv(covmat)
    # 2. compute distances for all runs and sum them
    sum_matrix = np.zeros((len(vectors[0]), len(vectors[0])))
    for run in vectors:
        vectors_mat = np.asarray(run)
        covmat = np.cov(np.transpose(vectors_mat))
        invcovmat = np.linalg.inv(covmat)
        matrix = np.zeros((len(run), len(run)))
        for i, vec1 in enumerate(run):
            for j, vec2 in enumerate(run):
                matrix[i, j] = np.matmul(np.transpose(vec1-vec2), np.matmul(invcovmat, (vec1-vec2)))
        sum_matrix += matrix
    return sum_matrix/len(vectors)


def rdm_crappynobis(vectors):
    # Normalize according to the stdev of each individual unit, then compute the euclidian distance.
    # THIS IS NOT REALLY MAHALANOBIS BECAUSE IT MAKES USE OF VARIANCE BUT NOT COVARIANCE.
    vectors_mat = np.asarray(vectors)
    zvecs = stats.zscore(vectors_mat, axis=1)
    matrix = np.zeros((len(vectors), len(vectors)))
    for i in range(len(zvecs)):
        for j in range(len(zvecs)):
            matrix[i, j] = np.sqrt(np.sum((zvecs[i, :]-zvecs[j, :])**2))
    return matrix

def plot_rdm(matrix, labels, title, show_rdm=False, vmin=None, vmax=None):
    if vmin is None and vmax is None:
        sns.heatmap(matrix, cbar=True, square=True, xticklabels=labels, yticklabels=labels)
    else:
        sns.heatmap(matrix, cbar=True, square=True, xticklabels=labels, yticklabels=labels, vmin = vmin, vmax = vmax)
    plt.title(title)
    if show_rdm:
        plt.show()

def compare_matrices(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        raise ValueError("both matrices must be the same shape")
    return stats.spearmanr(matrix1.flatten(), matrix2.flatten())[0]


def compare_matrices_person(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        raise ValueError("both matrices must be the same shape")
    return stats.pearsonr(matrix1.flatten(), matrix2.flatten())[0]

# Do activation distributions differences predict matrix differences
def kolmogorov_smirnov(sample1, sample2):
        return stats.ks_2samp(sample1, sample2)[0]