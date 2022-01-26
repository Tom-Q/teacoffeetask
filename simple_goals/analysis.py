import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn import discriminant_analysis as sklda
import utils
#import rsatoolbox

import seaborn as sns
import warnings

PEARSON = "pearson"
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

_mds_colors = ['b', 'g', 'r', 'c', 'm', 'y']
_mds_linestyles = ['-', '--', '-.', ':']
_mds_markerstyles = ['o', '|', 'x', '_']
# Now turn that into a list such that consecutive elements are as different from each other as possible
_mds_styles = []
for markerstyle in _mds_markerstyles:
    for linestyle in _mds_linestyles:
        for color in _mds_colors:
            _mds_styles.append(color+markerstyle+linestyle)

def make_mds(rdm, name, labels, title, pattern):
    mdsy = mds(rdm)
    idx = 0
    for style_idx, p in enumerate(pattern):
        plot_mds_points(mdsy[idx:idx+p], range(p), labels=labels[idx:idx+p], style=_mds_styles[style_idx])
        idx += p
    plt.title(title)
    plt.savefig(name + '_mds')
    plt.clf()

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

import scipy.spatial as scispa
def rdm_mahalanobis(vectors):
    vectors_mat = np.asarray(vectors)
    #mu = np.mean(vectors_mat)
    covmat = np.cov(vectors_mat.T)
    invcovmat = np.linalg.inv(covmat)
    matrix = np.zeros((len(vectors), len(vectors)))
    for i, vec1 in enumerate(vectors):
        for j, vec2 in enumerate(vectors):
            matrix[i, j] = scispa.distance.mahalanobis(vec1, vec2, invcovmat)
            # np.matmul(np.transpose(vec1-vec2), np.matmul(invcovmat, (vec1-vec2)))
    return np.nan_to_num(matrix)


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

def plot_rdm(matrix, labels, title, show_rdm=False, vmin=None, vmax=None, figsize=None, fontsize=None):
    if fontsize is not None:
        sns.set(font_scale=fontsize)
    if figsize is not None:
        plt.figure(figsize = (figsize, figsize))
    if vmin is None and vmax is None:
        plot = sns.heatmap(matrix, cbar=True, square=True, xticklabels=labels, yticklabels=labels)
    else:
        plot = sns.heatmap(matrix, cbar=True, square=True, xticklabels=labels, yticklabels=labels, vmin = vmin, vmax = vmax)

    # Basically I'll use a small font scale if the plot is huge.
    # Conversely that means the title is no longer visible.
    if fontsize is not None:
        plot.axes.set_title(title, fontsize=25/fontsize)
    else:
        plot.axes.set_title(title)
    #plot.set_xlabel("X Label", fontsize=20)
    #plot.set_ylabel("Y Label", fontsize=20)
    #plt.title(title)
    plt.tight_layout()
    if show_rdm:
        plt.show()


def save_rdm(rdm, labels, filename, title="", image=True, csv=True):
    if csv:
        np.savetxt(filename+'.txt', rdm, delimiter="\t", fmt='%.2e')
    if image:
        plot_rdm(rdm, labels, title, figsize=10, fontsize=0.6)
        plt.savefig(filename+'.png', dpi=300, bbox_inches='tight')


def compare_matrices_spearman(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        raise ValueError("both matrices must be the same shape")
    return stats.spearmanr(matrix1.flatten(), matrix2.flatten())[0]


def compare_matrices_pearson(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        raise ValueError("both matrices must be the same shape")
    return stats.pearsonr(matrix1.flatten(), matrix2.flatten())[0]

# Generate an RDM based on a list of RDMs
def rdm_of_rdms(rdms, type=SPEARMAN):
    numrdms = len(rdms)
    rdm = np.zeros((numrdms, numrdms))
    for i in range(numrdms):
        for j in range(i, numrdms):
            if type == SPEARMAN:
                rdm[i, j] = 1. - compare_matrices_spearman(rdms[i], rdms[j])
            elif type == PEARSON:
                rdm[i, j] = 1. - compare_matrices_pearson(rdms[i], rdms[j])
            else:
                raise NotImplementedError("unknown RDM comparison type")
            rdm[j, i] = rdm[i, j]
    return rdm

# Do activation distributions differences predict matrix differences
def kolmogorov_smirnov(sample1, sample2):
        return stats.ks_2samp(sample1, sample2)[0]


def rdm_ldt(data, noise=0.):  # Our data is an mxnxk matrix. m = samples, n = states, k = activation.
    m, n, k = data.shape
    # add noise to the data
    data += np.random.normal(loc=0., scale=noise, size=data.shape)
    if m % 2 != 0:
        # discard last sample
        data = data[:-1]
        m = m-1
        warnings.warn("for ldt we need an even number of samples. Discarding one sample")

    # Divide the data into two separate sets.
    set1 = data[:m//2]
    set2 = data[m//2:]

    # Run linear discriminant analysis for each pair of states...
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            # Get sample activations for 2 states from the training set
            train_state1 = set1[:, i, :].reshape(m//2, k)
            train_state2 = set1[:, j, :].reshape(m//2, k)
            # stack them:
            X = np.concatenate((train_state1, train_state2), axis=0)
            # give state 1 label "0" and state 2 label "1"
            y = np.hstack((np.zeros(m//2), np.ones(m//2)))
            # fit linear discriminant analysis
            lda = sklda.LinearDiscriminantAnalysis(solver='svd')
            lda.fit(X, y)

            #print("test")
            # save the intercept.
            coeffs = lda.coef_
            intercept = lda.intercept_

            # Get the same states from the test set
            test_state1 = set2[:, i, :].reshape(m//2, k)
            test_state2 = set2[:, j, :].reshape(m//2, k)
            # Compute "distance" (orthogonal vector value) with the intercept.
            # These "distances" (not really) will be positive or negative depending on the category
            distances_state1 = (np.dot(coeffs, np.transpose(test_state1)).reshape(-1) + intercept) / np.sqrt(np.sum(coeffs**2))
            distances_state2 = (np.dot(coeffs, np.transpose(test_state2)).reshape(-1) + intercept) / np.sqrt(np.sum(coeffs**2))

            # Now do a t-test to see if the two categories are separated.
            tvalue, p_value = stats.ttest_ind(distances_state1, distances_state2)
            distance = np.abs(tvalue)
            # This is our distance value for the RDM!
            rdm[i, j] = distance
            # Do the other side of the RDM, it's symmetrical.
            rdm[j, i] = distance

    # Fill the diagonal with 0s.
    for i in range(n):
        rdm[i, i] = 0.

    return rdm


from matplotlib.ticker import FuncFormatter
def barplot_figure_errors(filename):
    # goals
    means_action_errors = (7.13, 37.42, 62.66, 75.52, 80.81)
    means_subseq_errors = (12.47, 22.73, 20.79, 17.64, 15.51)

    # no goals
    #means_action_errors = (6.25, 35.41, 59.96, 75.15, 81.25)
    #means_subseq_errors = (8.57, 18.64, 19.22, 16.22, 14.44)


    ind = [1, 2, 3, 4, 5]  # the x locations for the groups
    width = 0.35
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(ind, means_subseq_errors, width, color='cornflowerblue')
    ax.bar(ind, means_action_errors, width, bottom=means_subseq_errors, color='orange')
    ax.bar(ind, [100]*len(means_subseq_errors), width, bottom=[a+b for (a, b) in zip(means_action_errors, means_subseq_errors)], color='bisque')
    ax.set_ylabel('Outcomes')
    ax.set_xlabel('Noise magnitude')
    #ax.set_title('Percentage of sequences displaying action or subsequence errors')
    plt.xticks(ind, ('1', '2', '3', '4', '5'))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    formatter = FuncFormatter(lambda y, pos: "%d%%" % (y))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_ylim([0, 100])
    ax.legend(labels=['Subsequence error', 'Action error', 'Success'], loc="upper left")
    plt.tight_layout()
    #plt.show()
    #plt.title(title)
    plt.savefig(filename)
    plt.clf()


def barplot_figure_ablations(filename):
    # Observation, Actions, Goal1, Goal2, Goal1 & 2,
    means_action_errors = (67.48, 58.29, 18.10, 3.57, 5.10)
    means_subseq_errors = (32.48, 22.57, 61.90, 57.10, 7.62)

    # no goals
    #means_action_errors = (6.25, 35.41, 59.96, 75.15, 81.25)
    #means_subseq_errors = (8.57, 18.64, 19.22, 16.22, 14.44)


    ind = [1, 2, 3, 4, 5]  # the x locations for the groups
    width = 0.35
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(ind, means_subseq_errors, width, color='orange')
    ax.bar(ind, means_action_errors, width, bottom=means_subseq_errors, color='cornflowerblue')
    ax.bar(ind, [100]*len(means_subseq_errors), width, bottom=[a+b for (a, b) in zip(means_action_errors, means_subseq_errors)], color='bisque')
    ax.set_ylabel('Outcomes')
    ax.set_xlabel('Units lesioned')
    #ax.set_title('Percentage of sequences displaying action or subsequence errors')
    plt.xticks(ind, ['Observations', 'Actions', 'Goals & subgoals', 'Subgoals only', 'Goals only'], rotation=20)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    formatter = FuncFormatter(lambda y, pos: "%d%%" % (y))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_ylim([0, 100])
    ax.legend(labels=['Subsequence error', 'Action error', 'Success'], loc="upper right")
    plt.tight_layout()
    #plt.show()
    #plt.title(title)
    plt.savefig(filename)
    plt.clf()

from matplotlib.patches import Patch
def bargraph_with_without_goalunits(filename):
    # no goals
    means_action_errors_goals = (6.25, 35.41, 59.96, 75.15, 81.25)
    means_subseq_errors_goals = (8.57, 18.64, 19.22, 16.22, 14.44)

    # goals
    means_action_errors_nogoals = (7.13, 37.42, 62.66, 75.52, 80.81)
    means_subseq_errors_nogoals = (12.47, 22.73, 20.79, 17.64, 15.51)

    ind = [1., 2., 3., 4., 5.]  # the x locations for the groups
    width = 0.2
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    idxs_goals = [id-0.11 for id in ind]
    plt.rcParams['hatch.color'] = 'white'  # using "edgecolor" causes a white edge to appear in between the different bars.
    ax.bar(idxs_goals, means_subseq_errors_goals, width, color='orangered')
    ax.bar(idxs_goals, means_action_errors_goals, width, bottom=means_subseq_errors_goals, color='orange')
    ax.bar(idxs_goals, [100]*len(means_subseq_errors_goals), width, bottom=[a+b for (a, b) in zip(means_action_errors_goals, means_subseq_errors_goals)], color='bisque')

    idxs_nogoals = [id+0.11 for id in ind]
    ax.bar(idxs_nogoals, means_subseq_errors_nogoals, width, color='orangered', hatch='////')
    ax.bar(idxs_nogoals, means_action_errors_nogoals, width, bottom=means_subseq_errors_nogoals, color='orange', hatch='////')
    ax.bar(idxs_nogoals, [100] * len(means_subseq_errors_goals), width, bottom=[a + b for (a, b) in zip(means_action_errors_nogoals, means_subseq_errors_nogoals)], color='bisque', hatch='////')
    ax.set_ylabel('Outcomes')
    ax.set_xlabel('Noise magnitude')
    #ax.set_title('Percentage of sequences displaying action or subsequence errors')
    plt.xticks(ind, ('1', '2', '3', '4', '5'))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    formatter = FuncFormatter(lambda y, pos: "%d%%" % (y))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_ylim([0, 100])
    legend_elements = [Patch(facecolor='orangered', label='Subsequence error'),
                       Patch(facecolor='orange', label='Action error'),
                       Patch(facecolor='bisque', label='Correct'),
                       #Patch(facecolor='white', edgecolor='grey', label='Goals'),
                       Patch(facecolor='grey', edgecolor='white', hatch='////', label='Flat network')]

    ax.legend(handles=legend_elements, loc='upper left')

    #ax.legend(labels=['Subsequence error', 'Action error', 'Success'], loc="upper left")
    plt.tight_layout()
    #plt.show()
    #plt.title(title)
    plt.savefig(filename)
    plt.clf()

def noise_plot_matplotlib():
    # prenoise, noise, noise+1, noise+2, noise+3, noise+4, error-4, error-3, error-2, error-1, error
    noise1 = [4.72, 8.22, 8.93, 10.07, 10.82, 12.45, 14.43, 16.31, 16.05, 14.70, 16.40]