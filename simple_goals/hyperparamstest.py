from neural import neuralnet
import utils
from pnas import pnas2018
import numpy as np
import analysis
import matplotlib.pyplot as plt
from scipy import stats


# Train 100 networks of each
def full_hyperparameter_test():
    counter = 0
    failed_networks = 0
    report = open("report.txt", 'a')
    report.write("\nSTART\n")
    for algorithm in [neuralnet.SGD, neuralnet.RMSPROP, neuralnet.ADAM]:
        for initialization in [neuralnet.NORMAL, neuralnet.UNIFORM]:
            for hidden_units in [8, 15, 50]:
                for l1reg in [0., 0.001]:
                    for l2reg in [0., 0.001]:
                        for loss_type in [pnas2018.CROSS_ENTROPY, pnas2018.MSE]:
                            learning_rate = 0.1
                            if algorithm == neuralnet.RMSPROP or algorithm == neuralnet.ADAM:
                                learning_rate *= 0.1
                            if hidden_units == 8:
                                learning_rate *= 2
                            elif hidden_units == 50:
                                learning_rate *= 0.5
                            elif loss_type == pnas2018.MSE and (algorithm != neuralnet.RMSPROP and algorithm != neuralnet.ADAM):
                                learning_rate *= 10
                            counter+=1
                            name = algorithm+initialization+str(hidden_units)+str(l1reg)+str(l2reg)+loss_type+"{:.3f}".format(learning_rate)
                            print("Hyperparameter test:" + name + '(' + str(counter) + ' out of 144)')
                            num_networks = 0
                            while num_networks < 50:
                                print(num_networks)
                                model, rng_avg_sequence = pnas2018.train(iterations=5000, algorithm=algorithm, size_hidden=hidden_units,
                                                       learning_rate=learning_rate, l1reg=l1reg, l2reg=l2reg,
                                                       loss_type=loss_type, initialization=initialization)
                                _, accuracy_totals = pnas2018.accuracy_test(model)
                                total_iterations = 5000
                                step = 5000
                                while (rng_avg_sequence < 0.22 or not np.array_equal(accuracy_totals, np.asarray([0.5, 0.5, 1, 1, 1, 1])))\
                                      and total_iterations < 10000:
                                    model, rng_avg_sequence = pnas2018.train(model, iterations=step, algorithm=algorithm,
                                                           size_hidden=hidden_units,
                                                           learning_rate=learning_rate, l1reg=l1reg, l2reg=l2reg,
                                                           loss_type=loss_type, initialization=initialization)
                                    _, accuracy_totals = pnas2018.accuracy_test(model)
                                    report.write(name + " extra iterations:"+str(step) + '\n')
                                    total_iterations += step
                                    print("total iterations="+str(total_iterations))
                                if np.array_equal(accuracy_totals, np.asarray([0.5, 0.5, 1, 1, 1, 1])):
                                    num_networks+=1
                                else:
                                    print("-----------------------------------------!!!!!!!!!!!!!!NETWORK FAILED!!!!!!!!!!!!!!!----------------------------------------")
                                    report.write(name + " failed\n")
                                    failed_networks += 1

                                utils.save_object(algorithm+initialization+str(hidden_units)+str(l1reg)+str(l2reg) + loss_type, model)
    print(failed_networks)
    report.close()


def hyperparameter_analysis(type=analysis.SPEARMAN, file="hyperparameter_test_rdm.txt"):
    # For each model: generate spearman matrix. Store with a label.
    count = 0
    matrices = []
    for hidden_units in [8, 15, 50]:
        for loss_type in [pnas2018.CROSS_ENTROPY, pnas2018.MSE]:
            for l1reg in [0., 0.001]:
                for l2reg in [0., 0.001]:
                    for algorithm in [neuralnet.SGD, neuralnet.RMSPROP, neuralnet.ADAM]:
                        for initialization in [neuralnet.NORMAL, neuralnet.UNIFORM]:
                                learning_rate = 0.1
                                if algorithm == neuralnet.RMSPROP or algorithm == neuralnet.ADAM:
                                    learning_rate *= 0.1
                                if hidden_units == 8:
                                    learning_rate *= 2
                                elif hidden_units == 50:
                                    learning_rate *= 0.5
                                elif loss_type == pnas2018.MSE and (
                                        algorithm != neuralnet.RMSPROP and algorithm != neuralnet.ADAM):
                                    learning_rate *= 10
                                name = algorithm + initialization + str(hidden_units) + str(l1reg) + str(
                                    l2reg) + loss_type# + "{:.3f}".format(learning_rate)
                                # Generate RDM and hidden activations
                                matrix, hidden_activation = pnas2018.make_rdm_multiple(name, 10, save_files=False, rdm_type=type)
                                hidden_activation = np.concatenate(hidden_activation, axis=None).flatten()
                                # make a graph for this networ
                                """
                                plt.hist(hidden_activation, bins=100, range=(0., 1.), rwidth=1, facecolor='blue', density=True)
                                plt.xlabel('activation')
                                plt.ylabel('units activating at that level')
                                plt.title('density plot hidden unit activation: '+name)
                                plt.grid(True)
                                plt.tight_layout()
                                #plt.show()
                                #plt.savefig(name.replace('.', '_') + '.png')
                                plt.clf()
                                """
                                matrices.append([name, matrix, hidden_activation])

    # Make an average of all the RDMs
    # mean_mat = np.mean([matrix for name, matrix in matrices], axis=0)
    # Save that rdm
    # np.savetxt("mean_mat.txt", mean_mat, delimiter="\t", fmt='%.2e')

    # Compare all spearman matrix. Generate a massive matrix.
    spearman_mat = np.zeros((len(matrices), len(matrices)))
    for i, mat1 in enumerate(matrices):
        for j, mat2 in enumerate(matrices):
            spearman_mat[i, j] = analysis.compare_matrices_spearman(mat1[1], mat2[1])

    # Also compare all hidden activations, generate massive matrix.
    activation_mat = np.zeros((len(matrices), len(matrices)))
    for i, mat1 in enumerate(matrices):
        for j, mat2 in enumerate(matrices):
            activation_mat[i, j] = analysis.kolmogorov_smirnov(mat1[2], mat2[2])

    # Now compare the two top level matrices: pearson
    # Only keep the upper triangle when pearson-correlating.
    spearman_mat_indices = spearman_mat[np.triu_indices_from(spearman_mat, k=1)]
    activation_mat_indices = activation_mat[np.triu_indices_from(activation_mat, k=1)]
    print(stats.pearsonr(spearman_mat_indices, activation_mat_indices))
    print(stats.spearmanr(spearman_mat_indices, activation_mat_indices))
    # Make a giant rdm with labels
    #labels = [mat[0] for mat in matrices]
    #analysis.plot_rdm(spearman_mat, labels, "Hyperparameter test general spearman matrix", vmin=-0.5, vmax=1.)
    #np.savetxt(file, spearman_mat, delimiter="\t", fmt='%.2e')
    #plt.tight_layout()
    #plt.savefig(file[:-3] + ".png")


def reload(file="hyperparameter_test_rdm.txt"):
    characteristics = []
    for hidden_units in [8, 15, 50]:
        for loss_type in [pnas2018.CROSS_ENTROPY, pnas2018.MSE]:
            for l1reg in [0., 0.001]:
                for l2reg in [0., 0.001]:
                    for algorithm in [neuralnet.SGD, neuralnet.RMSPROP, neuralnet.ADAM]:
                        for initialization in [neuralnet.NORMAL, neuralnet.UNIFORM]:
                            learning_rate = 0.1
                            if algorithm == neuralnet.RMSPROP or algorithm == neuralnet.ADAM:
                                learning_rate *= 0.1
                            if hidden_units == 8:
                                learning_rate *= 2
                            elif hidden_units == 50:
                                learning_rate *= 0.5
                            elif loss_type == pnas2018.MSE and (
                                    algorithm != neuralnet.RMSPROP and algorithm != neuralnet.ADAM):
                                learning_rate *= 10
                            if l1reg == 0.001:
                                L1 = 'l1'
                            else:
                                L1 = ''
                            if l2reg == 0.001:
                                L2 = 'l2'
                            else:
                                L2 = ''
                            if l1reg == l2reg == 0.:
                                L1 = 'no_reg'
                            name = str(hidden_units) + ' units, ' + loss_type + ', ' + 'LR:{:.2f}'.format(learning_rate) + ', ' + L1 + L2 + ', ' + algorithm + ', ' + initialization # + "{:.3f}".format(learning_rate)
                            characteristics.append([name, hidden_units, loss_type, learning_rate, l1reg, l2reg, algorithm, initialization])

    matrix = np.loadtxt(file, delimiter="\t")
    print(sort_quality(matrix))
    rows = matrix.tolist()
    # Now sort. Ugliest python line ever, what it does is this:
    # 1. Zip characteristics and rows, so that each row is paired with its characteristics in a tuple
    # 2. Sort this list of tuple for multiple conditions, using the characteristics in the order given
    # 3. Retrieve the sorted characteristics and rows separately
    sorted_stuff = [(sorted_chars, row) for sorted_chars, row in sorted(zip(characteristics, rows),
                                                key=lambda x: (x[0][1], x[0][2], x[0][3], x[0][4], x[0][5], x[0][6], x[0][7]))]
    sorted_rows = [row for (sorted_chars, row) in sorted_stuff]
    sorted_characteristics = [sorted_chars for (sorted_chars, row) in sorted_stuff]
    # Now transpose and sort again
    columns = np.asarray(sorted_rows).transpose().tolist()
    sorted_columns = [col for _, col in sorted(zip(characteristics, columns),
                                             key=lambda x: (x[0][1], x[0][2], x[0][3], x[0][4], x[0][5], x[0][6], x[0][7]))]

    matrix = np.asarray(sorted_columns)
    labels = [char[0] for char in sorted_characteristics]

    # Metric for sort quality: how much difference is there between consecutive values
    print(sort_quality(matrix))

    plt.figure(dpi=200)
    plt.subplots(figsize=(45, 30))
    analysis.plot_rdm(matrix, labels, "Hyperparameter test general spearman matrix", vmin=-0.5, vmax=1.)
    plt.tight_layout()
    plt.savefig(file[:-3] + ".png", dpi=300)

def sort_quality(rdm):
    total = 0
    for i in range(len(rdm)):
        for j in range(len(rdm)-1):
            total += np.abs(rdm[i, j]-rdm[i,j+1])
    return total


#names = ["sgdnormal150.00.0cross_entropy", "adamuniform500.0010.001mse", "rmspropnormal150.00.0cross_entropy"]
def hyperparameter_individual_matrix_analysis(names):
    # For each model: generate spearman matrix. Store with a label.
    count = 0
    matrices = []
    for name in names:
        # Generate RDM and hidden activations
        matrix, hidden_activation = pnas2018.make_rdm_multiple(name, 50, save_files=True)
        matrices.append([name, matrix])

    # Compare all spearman matrix. Generate a massive matrix.
    spearman_mat = np.zeros((len(matrices), len(matrices)))
    for i, mat1 in enumerate(matrices):
        for j, mat2 in enumerate(matrices):
            spearman_mat[i, j] = analysis.compare_matrices_spearman(mat1[1], mat2[1])

    print(spearman_mat)

    # Make a giant rdm with labels
    #labels = [mat[0] for mat in matrices]
    #analysis.plot_rdm(spearman_mat, labels, "Hyperparameter test general spearman matrix", vmin=-0.5, vmax=1.)
    #np.savetxt(file, spearman_mat, delimiter="\t", fmt='%.2e')
    #plt.tight_layout()
    #plt.savefig(file[:-3] + ".png")

"""
# Simulations: extra variations
# 1. Standard: SGD
# 2. Algorithm:
# 2. a. RMSPROP
for i in range(0):
    print(i + 1)
    model = pnas2018.train(iterations=5000, algorithm=neuralnet.RMSPROP)
    pnas2018.accuracy_test(model)
    utils.save_object("extra2a_rmsprop", model)
#pnas2018.make_rdm_multiple("extra2a_rmsprop", 1, title="extra 2a: RMSPROP")

# 2. b. ADAM
for i in range(0):
    print(i + 1)
    model = pnas2018.train(iterations=5000, algorithm=neuralnet.ADAM)
    pnas2018.accuracy_test(model)
    utils.save_object("extra2b_adam", model)
#pnas2018.make_rdm_multiple("extra2b_adam", 100, title="extra 2b: ADAM")

# 3. Regularization
# 3. a. L1
for i in range(0):
    print(i + 1)
    model = pnas2018.train(iterations=5000, algorithm=neuralnet.SGD)
    pnas2018.accuracy_test(model)
    utils.save_object("extra3a_l1reg", model)
#pnas2018.make_rdm_multiple("extra3a_l1reg", 100, title="extra 3a: l1reg")
# 3. b. L2
for i in range(0):
    print(i + 1)
    model = pnas2018.train(iterations=5000, algorithm=neuralnet.SGD)
    pnas2018.accuracy_test(model)
    utils.save_object("extra3b_l2reg", model)
#pnas2018.make_rdm_multiple("extra3b_l2reg", 100, title="extra 3b: l2reg")

# 4. Learning rate
# 4. a. x5 ?
# 4. b. /5 ?
# 5. Learning length
# 5. a. x5 ?
# 5. b. /5 ?
# 6. Size hidden layer
# 6. a. x3
# 6. b. /3
# 7. Everything at once:
# 7a. RMSPROP + L1 + L2 + x5 learning rate + /5 learning length + 50 hidden units

#TODO: just l1, just l2; rms50; invert MSE/CE; check out all that; make a dissimilarity matrix of the factors

# That worked - reduce l2reg
for i in range(0):
    print(i + 1)
    model = pnas2018.train(iterations=5000, algorithm=neuralnet.RMSPROP, size_hidden=8, learning_rate=0.03, l1reg=0.0, l2reg=0.001, loss_type=pnas2018.CROSS_ENTROPY)
    pnas2018.accuracy_test(model)
    utils.save_object("rmsprop_8_units_l10_l20001_0", model)
#rms8l2 = pnas2018.make_rdm_multiple("rmsprop_8_units_l10_l20001_0", 100, title="RMSPROP/hidden=8/lrate=0.03/l1=0.0/l2=0.0001/loss=CE/init=0")


for i in range(0):
    print(i + 1)
    model = pnas2018.train(iterations=5000, algorithm=neuralnet.RMSPROP, size_hidden=8, learning_rate=0.03, l1reg=0.0, l2reg=0.001, loss_type=pnas2018.CROSS_ENTROPY)
    pnas2018.accuracy_test(model)
    utils.save_object("rmsprop_8_units_l10_l20001_0", model)
#rms8l2 = pnas2018.make_rdm_multiple("rmsprop_8_units_l10_l20001_0", 100, title="RMSPROP/hidden=8/lrate=0.03/l1=0.0/l2=0.0001/loss=CE/init=0")


for i in range(0):
    print(i + 1)
    model = pnas2018.train(iterations=5000, algorithm=neuralnet.RMSPROP, size_hidden=50, learning_rate=0.01, l1reg=0.0, l2reg=0.001, loss_type=pnas2018.CROSS_ENTROPY)
    pnas2018.accuracy_test(model)
    utils.save_object("rmsprop_50_units_l10_l20001_0", model)
#rms50l2 = pnas2018.make_rdm_multiple("rmsprop_50_units_l10_l20001_0", 100, title="RMSPROP/hidden=50/lrate=0.01/l1=0.0/l2=0.001/loss=CE/init=0")

for i in range(0):
    print(i + 1)
    model = pnas2018.train(iterations=5000, algorithm=neuralnet.RMSPROP, size_hidden=50, learning_rate=0.01, l1reg=0.001, l2reg=0.0, loss_type=pnas2018.CROSS_ENTROPY)
    pnas2018.accuracy_test(model)
    utils.save_object("rmsprop_50_units_l10001_l20_0", model)
#rms50l1 = pnas2018.make_rdm_multiple("rmsprop_50_units_l10001_l20_0", 100, title="RMSPROP/hidden=50/lrate=0.01/l1=0.001/l2=0.0/loss=CE/init=0")


# That worked
# 7b. ADAM + L1 + L2 + /5 learning rate + *5 learning length + 5 hidden units + random initialization
for i in range(0):
    print(i + 1)
    model = pnas2018.train(iterations=5000, algorithm=neuralnet.ADAM, size_hidden=50, learning_rate=0.003, l1reg=0.001, l2reg=0.0, loss_type=pnas2018.MSE)
    pnas2018.accuracy_test(model)
    utils.save_object("adam_50_units_l10001_l20_0", model)
#adam50l1 = pnas2018.make_rdm_multiple("adam_50_units_l10001_l20_0", 100, title="ADAM/hidden=50/lrate=0.01/l1=0.001/l2=0.0/loss=MSE/init=0")

# that worked
# 7b. ADAM + L1 + L2 + /5 learning rate + *5 learning length + 5 hidden units + random initialization
for i in range(0):
    print(i + 1)
    model = pnas2018.train(iterations=5000, algorithm=neuralnet.ADAM, size_hidden=8, learning_rate=0.01, l1reg=0.0, l2reg=0.001, loss_type=pnas2018.MSE)
    pnas2018.accuracy_test(model)
    utils.save_object("adam_8_units_l10_l2001_0", model)
#adam8l2 = pnas2018.make_rdm_multiple("adam_8_units_l10_l2001_0", 100, title="ADAM/hidden=50/lrate=0.01/l1=0.0/l2=0.001/loss=MSE/init=0")

for i in range(0):
    print(i + 1)
    model = pnas2018.train(iterations=5000, algorithm=neuralnet.SGD, size_hidden=15, learning_rate=0.1, l1reg=0.001, l2reg=0.0, loss_type=pnas2018.CROSS_ENTROPY)
    pnas2018.accuracy_test(model)
    utils.save_object("sgd_15_units_l10001_l20_0", model)
#controll1 = pnas2018.make_rdm_multiple("sgd_15_units_l10001_l20_0", 100, title="SGD/hidden=15/lrate=0.1/l1=0./l2=0.0/loss=CE/init=0")

for i in range(0):
    print(i + 1)
    model = pnas2018.train(iterations=5000, algorithm=neuralnet.SGD, size_hidden=50, learning_rate=0.05, l1reg=0.0, l2reg=0.0, loss_type=pnas2018.CROSS_ENTROPY)
    pnas2018.accuracy_test(model)
    utils.save_object("sgd_50_units_l10_l20_0", model)
#control50 = pnas2018.make_rdm_multiple("sgd_50_units_l10_l20_0", 100, title="SGD/hidden=50/lrate=0.05/l1=0./l2=0.0/loss=CE/init=0")

for i in range(0):
    print(i + 1)
    model = pnas2018.train(iterations=5000, algorithm=neuralnet.SGD, size_hidden=50, learning_rate=0.05, l1reg=0.001, l2reg=0.0, loss_type=pnas2018.CROSS_ENTROPY)
    pnas2018.accuracy_test(model)
    utils.save_object("sgd_50_units_l10001_l20_0", model)
#control50l1 = pnas2018.make_rdm_multiple("sgd_50_units_l10001_l20_0", 100, title="SGD/hidden=50/lrate=0.05/l1=0.001/l2=0.0/loss=CE/init=0")

#That worked
for i in range(0):
    print(i + 1)
    model = pnas2018.train(iterations=5000, algorithm=neuralnet.SGD, size_hidden=15, learning_rate=0.1, l1reg=0.0, l2reg=0.001, loss_type=pnas2018.CROSS_ENTROPY)
    pnas2018.accuracy_test(model)
    utils.save_object("sgd_15_units_l10_l20001_0", model)
#controll2 = pnas2018.make_rdm_multiple("sgd_15_units_l10_l20001_0", 100, title="SGD/hidden=15/lrate=0.1/l1=0./l2=0.0/loss=CE/init=0")

for i in range(100):
    print(i + 1)
    model = pnas2018.train(iterations=5000, algorithm=neuralnet.SGD, size_hidden=15, learning_rate=0.1, l1reg=0.0, l2reg=0.0, loss_type=pnas2018.CROSS_ENTROPY, initialization="uniform")
    pnas2018.accuracy_test(model)
    utils.save_object("sgd_15_units_uniform", model)

for i in range(0):
    print(i + 1)
    model = pnas2018.train(iterations=5000, algorithm=neuralnet.ADAM, size_hidden=50, learning_rate=0.003, l1reg=0.001, l2reg=0.0, loss_type=pnas2018.MSE, initialization="uniform")
    pnas2018.accuracy_test(model)
    utils.save_object("adam_50_units_l10001_l20_uniform", model)

for i in range(0):
    print(i + 1)
    model = pnas2018.train(iterations=5000, algorithm=neuralnet.RMSPROP, size_hidden=8, learning_rate=0.03, l1reg=0.0, l2reg=0.001, loss_type=pnas2018.CROSS_ENTROPY, initialization="uniform")
    pnas2018.accuracy_test(model)
    utils.save_object("rmsprop_8_units_l10_l20001_uniform", model)

import analysis
#print(analysis.compare_matrices(rdmat1, rdmat2))
#print(analysis.compare_matrices(rdmat2, rdmat3))
#print(analysis.compare_matrices(rdmat1, rdmat3))
#sys.exit()

control = pnas2018.make_rdm_multiple("sgd_15_units_l10_l20_0", 100, title="SGD/hidden=15/lrate=0.1/l1=0./l2=0.0/loss=CE/init=0")
controll1 = pnas2018.make_rdm_multiple("sgd_15_units_l10001_l20_0", 100, title="SGD/hidden=15/lrate=0.1/l1=0./l2=0.0/loss=CE/init=0")
controll2 = pnas2018.make_rdm_multiple("sgd_15_units_l10_l20001_0", 100, title="SGD/hidden=15/lrate=0.1/l1=0./l2=0.0/loss=CE/init=0")
rms8l2 = pnas2018.make_rdm_multiple("rmsprop_8_units_l10_l20001_0", 100, title="RMSPROP/hidden=8/lrate=0.03/l1=0.0/l2=0.0001/loss=CE/init=0")
rms8l1l2 = pnas2018.make_rdm_multiple("rmsprop_8_units_l1001_l20001_0", 100, title="RMSPROP/hidden=8/lrate=0.03/l1=0.0001/l2=0.0001/loss=CE/init=0")
adam50l1l2 = pnas2018.make_rdm_multiple("adam_50_units_l1001_l2001_0", 100, title="ADAM/hidden=50/lrate=0.01/l1=0.001/l2=0.001/loss=MSE/init=0")
adam8l1l2 = pnas2018.make_rdm_multiple("adam_8_units_l1001_l2001_0", 100, title="ADAM/hidden=50/lrate=0.01/l1=0.001/l2=0.001/loss=MSE/init=0")
adam50l1 = pnas2018.make_rdm_multiple("adam_50_units_l1001_l20_0", 100, title="ADAM/hidden=50/lrate=0.01/l1=0.001/l2=0.0/loss=MSE/init=0")
adam8l2 = pnas2018.make_rdm_multiple("adam_8_units_l10_l2001_0", 100, title="ADAM/hidden=50/lrate=0.01/l1=0.0/l2=0.001/loss=MSE/init=0")
rms50l2 = pnas2018.make_rdm_multiple("rmsprop_50_units_l10_l20001_0", 100, title="RMSPROP/hidden=50/lrate=0.01/l1=0.0/l2=0.001/loss=CE/init=0")
rms50l1 = pnas2018.make_rdm_multiple("rmsprop_50_units_l10001_l20_0", 100, title="RMSPROP/hidden=50/lrate=0.01/l1=0.001/l2=0.0/loss=CE/init=0")
control50 = pnas2018.make_rdm_multiple("sgd_50_units_l10_l20_0", 100, title="SGD/hidden=50/lrate=0.05/l1=0./l2=0.0/loss=CE/init=0")
control50l1 = pnas2018.make_rdm_multiple("sgd_50_units_l10001_l20_0", 100, title="SGD/hidden=50/lrate=0.05/l1=0.001/l2=0.0/loss=CE/init=0")
controluniform = pnas2018.make_rdm_multiple("sgd_15_units_uniform", 100, title="SGD/hidden=15/lrate=0.1/l1=0.0/l2=0.0/loss=CE/uniform")
adam50l1uniform = pnas2018.make_rdm_multiple("sgd_15_units_uniform", 100, title="SGD/hidden=15/lrate=0.003/l1=0.001/l2=0.0/loss=MSE/uniform")
rms8l2uniform = pnas2018.make_rdm_multiple("sgd_15_units_uniform", 100, title="SGD/hidden=15/lrate=0.03/l1=0.0/l2=0.001/loss=CE/uniform")


all_models = [['control', control],
              ['controll1', controll1],
              ['controll2', controll2],
              ['control50', control50],
              ['control50l1', control50l1],
              ['rms8l1l2', rms8l1l2],
              ['rms8l2', rms8l2],
              ['rms50l1', rms50l1],
              ['rms50l2', rms50l2],
              ['adam50l1l2', adam50l1l2],
              ['adam8l1l2', adam8l1l2],
              ['adam50l1',  adam50l1],
              ['adam8l2', adam8l2]]

for i in all_models:
    for j in all_models:
        print(i[0] + ' ' + j[0] + '---' + str(analysis.compare_matrices(i[1], j[1])))
"""