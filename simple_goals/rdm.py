import numpy as np
import analysis
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import utils
import scipy
from scipy import stats, spatial
from sklearn import discriminant_analysis as sklda

# RDM object
RDM_COLOR_DIVERGING = "RDM COLOR DIVERGING"
RDM_COLOR_SEQUENTIAL = "RDM COLOR SEQUENTIAL"

PEARSON = "pearson"
SPEARMAN="spearman"
MAHALANOBIS = "mahalanobis"
EUCLIDIAN = "euclidian"
CRAPPYNOBIS = "crappynobis"

class rdm(object):
    def __init__(self, properties, matrix_values = None, vectors=None, type=None):
        """
        @param properties: Each property is a dictionary (key + value) describing one time step in the rdm.
        This enable sorting the RDM according to multiple criteria.
        @param matrix_values: EITHER numpy matrix of already-computed values, or list of VECTORS + TYPE
        @param vectors:
        @param type: type of matrix (euclidian or spearman)
        """
        if matrix_values is not None:
            self.matrix = matrix_values
        else:
            if type == SPEARMAN: self.matrix = rdm_spearman(vectors)
            elif type == EUCLIDIAN: self.matrix = rdm_euclidian(vectors)
            else: raise NotImplementedError("only euclidian and spearman please")

        self.properties = properties

    def delete_entry(self, idx):
        self.matrix = np.delete(self.matrix, idx, axis=0)
        self.matrix = np.delete(self.matrix, idx, axis=1)
        del self.properties[idx]

    def sort_by(self, *arguments):
        for arg in arguments:
            argument = arg[0]
            reverse = arg[1]
            # Sort the indexes, and get the indexes of the new positions.
            # E.g. sort(["b", "d", "c"]) should return ["b", "c", "d"] and [0, 2, 1].
            # Performance is not an issue, rdms don't get very large.
            # (1) Give properties an index for sorting
            for idx, p in enumerate(self.properties):
                p['_index'] = idx
            # (2) sort the properties based on whatever argument
            self.properties = sorted(self.properties, key=lambda d: str(d[argument]), reverse=reverse)
            # (3) extract the indices into a list.
            indices = np.array([p['_index'] for p in self.properties])
            # (4) sort the matrix accordingly in place.
            self._reorder_matrix(indices)


    def _reorder_matrix(self, new_order):
        # in place reordering using a copy matrix. Probably possible to
        # be twice as efficient but this does the trick neatly
        new_mat = self.matrix.copy()
        # Rows
        for old_ind, new_ind in enumerate(new_order):
            new_mat[old_ind, :] = self.matrix[new_ind, :]
        # Columns
        for old_ind, new_ind in enumerate(new_order):
            self.matrix[:, old_ind] = new_mat[:, new_ind]


    def get_labels(self):
        if 'label' in self.properties[0].keys():
            return [p['label'] for p in self.properties]
        else:
            labels = []
            for p in self.properties:
                label = ''
                for key, value in p.items():
                    if key != '_index' and key != 'label' and value is not None:
                        label += str(value) + '.'
                labels.append(label)
            return labels

    def save(self, filename, title="", labels=None, image=True, csv=True, figsize=None, fontsize=None,
             color=RDM_COLOR_SEQUENTIAL, dpi=300):
        if figsize is None:
            figsize = 10. * len(self.properties)/48.
        if fontsize is None:
            fontsize = 0.6 * len(self.properties)/48.
        if csv:
            np.savetxt(filename + '.txt', self.matrix, delimiter="\t", fmt='%.2e')
        if image:
            self.plot_rdm(title, labels, figsize=figsize, fontsize=fontsize, color=color)
            plt.savefig(filename + '.png', dpi=dpi, bbox_inches='tight')
            plt.clf()

    def plot_rdm(self, title, labels=None, show_rdm=False, vmin=None, vmax=None, figsize=None, fontsize=None,
                 color=RDM_COLOR_SEQUENTIAL):
        if labels is None:
            labels = self.get_labels()
        if fontsize is not None:
            sns.set(font_scale=fontsize)
        if figsize is not None:
            plt.figure(figsize=(figsize, figsize))

        if color == RDM_COLOR_DIVERGING:
            sns.color_palette("vlag", as_cmap=True)
        else:
            sns.color_palette("magma", as_cmap=True)

        if vmin is None and vmax is None:
            plot = sns.heatmap(self.matrix, cbar=True, square=True, xticklabels=labels, yticklabels=labels, center=0)
        else:
            plot = sns.heatmap(self.matrix, cbar=True, square=True, xticklabels=labels, yticklabels=labels, vmin=vmin,
                               vmax=vmax)

        # Basically I'll use a small font scale if the plot is huge.
        # Conversely that means the title is no longer visible.
        if fontsize is not None:
            plot.axes.set_title(title, fontsize=10/fontsize)
        else:
            plot.axes.set_title(title)
        # plot.set_xlabel("X Label", fontsize=20)
        # plot.set_ylabel("Y Label", fontsize=20)
        # plt.title(title)
        plt.tight_layout()
        if show_rdm:
            plt.show()

    def average_values(self, preserve_keys, ignore_func):
        """
        @param preserve_keys: differences we wish to preserve. Only average if these are the same.
        @param average_over_keys: differences we wish to abolish by averaging over them
        @param dont_cross_keys: function to identify elements to ignore while averaging
        """
        size_old_mat = len(self.properties)
        # 1. Use a hash, based on properties, to form element groups.
        # Elements are grouped together if they have the same preserved keys (same hash of preserved keys).
        # Elements that cross keys are simply discarded.
        # Each group (once averaged) corresponds to one member of the final matrix.
        groups = {}  # keys = hash made from properties. value = list of elements.
        count = 0
        dont_cross_count = 0
        for k in range(size_old_mat):
            for l in range(k+1, size_old_mat):
                element_row = k
                element_column = l

                # 2.a Test if the element is "crossed" (and therefore discarded).
                if ignore_func(self.properties[element_row], self.properties[element_column]):
                    dont_cross_count += 1
                    continue

                # 2. b also discard elements that would go straight to the diagonal (identical keys).
                group_key_r = ""
                group_key_c = ""
                for key in preserve_keys:
                    group_key_r += str(self.properties[element_row][key])
                    group_key_c += str(self.properties[element_column][key])
                if group_key_r == group_key_c:
                    continue

                # 2.c If not crossed, and not diagonal, make a hash key from the relevant properties,
                # and add to the corresponding group, or create it if needed
                sorted_keys = sorted([group_key_r, group_key_c])  #2A 1A is the same as 1A 2A. It feels like this shouldn't matter but it does!!
                group_key = sorted_keys[0] + sorted_keys[1]
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append((k, l))
                count += 1

        # Now we've got a list of the all the elements in the original matrix, grouped how they will contribute to the
        # final matrix. (Note: each element in the original matrix only contributes to one element in the final matrix).


        # 4. We need to reorganize the groups. To do this, we discover all the different columns, by generating them
        # from the "preserved" keys. For each column, we keep track of how many elements are in this column.
        # Since we're working with a half-RDM, the number of elements in a column correspond to the order of this
        # column in the RDM (starting at 0), like so:
        # 0 1 2
        # 0 a b
        # x 0 c
        # x x 0
        property_dict = {}
        for group in groups.values():
            group_elem = group[0]  # Any element of the group will do.
            for i in range(2):  # 0 = rows, 1 = columns
                new_property_key = ''
                for key in preserve_keys:
                    new_property_key += str(self.properties[group_elem[i]][key])

                if new_property_key not in property_dict:
                    new_property = {}
                    for key in preserve_keys:
                        new_property[key] = self.properties[group_elem[i]][key]
                    property_dict[new_property_key] = [new_property, 1 if i == 1 else 0]  # tuple with property list and count
                else:
                    if i == 1:  # Count the column entries
                        property_dict[new_property_key][1] += 1

        # 5. Fill in matrix. Now we can parse through every group, and from the group properties we can
        # identify the corresponding position in the final matrix using "property_dict".
        size_new_mat = len(property_dict)  #int(np.sqrt(len(groups)+.25)+.5)
        new_matrix = np.zeros((size_new_mat, size_new_mat))
        for group in groups.values():
            # Compute the group average
            average = 0
            for elem in group:
                average += self.matrix[elem[0], elem[1]]
            average /= len(group)

            # Make a key for the column and a key for the row.
            row_key = ""
            column_key = ""
            group_elem = group[0]
            for key in preserve_keys:
                row_key += str(self.properties[group_elem[0]][key])
                column_key += str(self.properties[group_elem[1]][key])

            row_idx = property_dict[row_key][1]
            column_idx = property_dict[column_key][1]
            new_matrix[row_idx, column_idx] = average

        # Sort the list of properties
        numbered_properties = list(property_dict.values())
        sorted_properties = [None]*len(numbered_properties)
        for numbered_property in numbered_properties:
            sorted_properties[numbered_property[1]] = numbered_property[0]

        # 7 Make all this into an RDM
        new_rdm = rdm(sorted_properties, matrix_values=new_matrix)

        # 8. Mirror the matrix over the diagonal
        for i in range(size_new_mat):
            for j in range(i, size_new_mat):
                if i == j:
                    new_rdm.matrix[i, j] = 0
                    continue
                else:
                    new_rdm.matrix[j, i] = new_rdm.matrix[i, j]

        return new_rdm

    def get_average_key(self, keys_values=None, equals=None, unequals=None, unequals_or=None): # Gets the average for a key and a value
        """
        Would be nice to have something more general but right now I only need equals/unequals/unequals_or, so no need to solve that little engineering problem in full
        @param keys_values: dictionaries of key/value conditions (average is made over only those entries for that key and that value)
        @param equals: list of list of keys that must be equal to each other. E.g. [[key1,key2], [key1, key3, key4]]: key1 must equal key2, 3, and 4.
        @param unequals: Same except here the values must be different
        @param unequals_or: At least one of the values must be different
        @return: A single average value for those distances
        """
        sum = 0.
        count = 0
        for i, prop_row in enumerate(self.properties):
            for j, prop_column in enumerate(self.properties):
                valid = True
                # Check that at least one has the correct key/values
                if keys_values is not None:
                    for key, value in keys_values.items():
                        if prop_row[key] != value and prop_column[key] != value:
                            valid = False
                            break
                    if not valid:
                        continue
                #Check that both have matching equalities
                if equals is not None:
                    for key in equals:
                        if prop_row[key] != prop_column[key]:
                            valid = False
                            break
                    if not valid:
                        continue
                # Check that unequalities are also correct
                if unequals is not None:
                    for key in unequals:
                        if prop_row[key] == prop_column[key]:
                            valid = False
                            break
                    if not valid:
                        continue
                # Check that "or" inequalities are also correct
                if unequals_or is not None:
                    valid = False
                    for key in unequals_or:
                        if prop_row[key] != prop_column[key]:
                            valid = True
                            break
                if not valid:
                    continue
                sum += self.matrix[i, j]
                count += 1
        if count == 0:
            return 0
        return sum/count

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
            matrix[i, j] = scipy.spatial.distance.mahalanobis(vec1, vec2, invcovmat)
            # np.matmul(np.transpose(vec1-vec2), np.matmul(invcovmat, (vec1-vec2)))
    return np.nan_to_num(matrix)

def rdm_noisy2_mahalanobis(vectors, noise=1., extra_samples=10):
    vectors_mat = np.asarray(vectors)
    vectors_mat2 = np.concatenate([vectors_mat+np.random.normal(loc=0., scale=noise, size=vectors_mat.shape) for i in range(extra_samples)], axis=0)
    #mu = np.mean(vectors_mat)
    covmat = np.cov(vectors_mat2.T)
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
