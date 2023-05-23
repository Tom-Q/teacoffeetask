import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import utils

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
#import rsatoolbox


#Multidimensional scaling
def mds(matrix):
    if matrix is list:
        # Make array from vectors
        matrix = np.stack(matrix, axis=0)

    # Set-up manifold methods
    mds = MDS(n_components=2, metric=True)
    # Plot results
    Y = mds.fit_transform(matrix)
    print(mds.stress_)
    return Y

def plot_mds_points(mds_y, points, labels=None, style='rx-', fontsize=9):
    toplotx = [mds_y[point, 0] for point in points]
    toploty = [mds_y[point, 1] for point in points]

    for i in range(len(toplotx)):
        x = toplotx[i]
        y = toploty[i]
        if labels == None:
            label = str(i)
        else:
            label = labels[i]
        plt.text(x * (1 + 0.01), y * (1 + 0.01), label, fontsize=fontsize)

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
    #means_action_errors = (81.48, 58.29, 18.10, 3.57, 5.10)
    #means_subseq_errors = (32.48, 22.57, 61.90, 57.10, 7.62)


    means_action_errors = (81.33, 71.05, 43.52, 7.14, 44.10)
    means_subseq_errors = (18.57, 18.86, 17.52, 55.71, 51.81)

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
    plt.xticks(ind, ['Observations', 'Actions', 'Subgoals', 'Goals', 'Subgoals and Goals'], rotation=20)
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

def bargraph_with_without_goalunits(filename):
    # Updated info based on 20 networks of each, 1 run each.
    # goals
    means_action_errors_goals = (0.02, 0.76, 5.36, 59.88, 93.51)
    means_subseq_errors_goals = (0.05, 2.33, 18.04, 33.38, 4.63)

    # no goals
    means_action_errors_nogoals = (0.07, 0.28, 5.72, 50.24, 90.12)
    means_subseq_errors_nogoals = (0.17, 0.93, 25.86, 43.62, 8.09)

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
    plt.xticks(ind, ('0.01', '0.1', '0.2', '0.5', '1.0'))
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

def loss_big_plot(filename="loss_plots"):
    fig = plt.figure()
    plt.clf()
    #plt.title('Loss per timestep between noise injection and error')
    plotcount = 0
    for length in [3, 6, 10]:
        for goals in [True, False]:
            plotcount += 1
            ax1 = fig.add_subplot(320 + plotcount)
            _loss_plot(ax1, length, 2, goals)
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

def _loss_plot(ax, length, noise_level, goals):
    if length == 3 and noise_level == 2:
        if goals:
            noise_actions = [3.28,	3.26,	3.26,	5.02]
            noise_goals = [2.69,	3.35,	3.89,	4.31]
        else:
            noise = [3.26,	3.30,	3.16,	5.17]
    elif length == 6 and noise_level == 2:
        if goals:
            noise_actions = [3.62, 3.67, 3.01, 3.75, 3.83, 4.74, 6.46]
            noise_goals = [3.22,	3.42,	4.93,	6.83,	7.73,	8.81,	10.47]
        else:
            noise = [3.76,	3.53,	3.11,	3.13,	3.23,	3.36,	5.25]
    elif length == 10 and noise_level == 2:
        if goals:
            noise_actions = [3.71,	3.41,	3.24,	2.85,	3.29,	3.03,	2.61,	3.43,	3.60,	4.12,	6.75]
            noise_goals = [3.84,	2.57,	3.32,	4.21,	3.41,	4.25,	6.00,	8.51,	8.82,	9.62,	11.64]
        else:
            noise = [4.91,	4.85,	5.10,	4.48,	4.87,	4.22,	3.64,	3.88,	4.25,	4.30,	6.11]

    ax.title.set_text('Error occurs on step ' + str(length))
    if goals:
        timestep = range(len(noise_actions))
        ax.plot(timestep, noise_goals, color='orangered', marker='+', label="Loss: goals")
        ax.plot(timestep, noise_actions, color='orange', marker='+', label="Loss: actions")
    else:
        timestep = range(len(noise))
        ax.plot(timestep, noise, color='orange', marker='+', label="Loss: actions")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_ticks(timestep)

    #ax.legend()
    ax.set_ylabel("Loss")
    ax.set_xlabel("Timesteps since noise injection")


def plot_tsne(filefrom, fileto):
    points = np.loadtxt(filefrom)
    points = points.T#reshape((2, -1))
    print(points)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    points_seq1 = points[:, 0:39]
    points_seq2 = points[:, 39:74]
    points_seq3 = points[:, 74:109]

    # plot seq 1
    ax.plot(points_seq1[0, :], points_seq1[1, :], color='orange', linestyle='solid', marker='+')

    # Plot seq 2
    ax.plot(points_seq2[0, :], points_seq2[1, :], color='brown', linestyle='dashed', marker='+')

    # Plot seq 3
    ax.plot(points_seq3[0, :], points_seq3[1, :], color='#79afdaff', linestyle='dotted', marker='+')

    # start points
    ax.plot(points_seq1[0, 0], points_seq1[1, 0], color='orange', marker='>')
    ax.plot(points_seq2[0, 0], points_seq2[1, 0], color='brown', marker='>')
    ax.plot(points_seq3[0, 0], points_seq3[1, 0], color='#79afdaff', marker='>')

    # End points
    ax.plot(points_seq1[0, -1], points_seq1[1, -1], color='orange', marker='o')
    ax.plot(points_seq2[0, -1], points_seq2[1, -1], color='brown', marker='o')
    ax.plot(points_seq3[0, -1], points_seq3[1, -1], color='#79afdaff', marker='o')

    ax.plot(points_seq3[0, 28:36], points_seq3[1, 28:36], linestyle='None', color='#79afdaff', marker='v', markerfacecolor='black')
    ax.plot(points_seq3[0, 14:23], points_seq3[1, 14:23], linestyle='None', color='#79afdaff', marker='v', markerfacecolor='white')

    legend_elements = [Line2D([0], [0], color='orange', label='Coffee cream', linestyle='solid'),
                       Line2D([0], [0], color='brown', label='Tea milk', linestyle='dashed'),
                       Line2D([0], [0], color='#79afdaff', label='Tea cream', linestyle='dotted'),
                       Line2D([0], [0], marker='>', color='w', label='Sequence start', markerfacecolor='k'),
                       Line2D([0], [0], marker='o', color='w', label='Sequence end', linestyle='', markerfacecolor='k'),
                       Line2D([0], [0], marker='v', linestyle='dotted', color='#79afdaff', label='Clamped: Tea', markerfacecolor='black'),
                       Line2D([0], [0], marker='v', linestyle='dotted', color='#79afdaff', label='Clamped: Cream', markerfacecolor='white'),
                       ]
    ax.legend(handles=legend_elements, bbox_to_anchor = (0, 0))#loc='lower left')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig(fileto, format='svg')
    plt.clf()


from numpy import genfromtxt
import seaborn as sns
import pandas as pd
def graph_error_rate_per_step_example(fileto="testfig.svg"):
    errors = genfromtxt("example_data.csv", delimiter=',')

    trials_per_step = np.zeros_like(errors)
    for net in range(50):
        remainders = 100
        for step in range(50):
            trials_per_step[net,step] = remainders
            remainders -= errors[net,step]

    errors = np.divide(errors, trials_per_step)
    errors *= 100
    errors = pd.DataFrame(errors, columns=["1: fixate cupboard", "2: open", "3: fixate coffee jar", "4: take", "5: open",
                          "6: fixate mug", "7: add to mug", "8: fixate coffee jar", "9: close",
                          "10: fixate cupboard", "11: put down", "12: fixate spoon", "13: take", "14: fixate mug",
                          "15: stir ", "16: fixate table ", "17: put down ", "18: fixate sugar box ", "19: take",
                          "20: fixate mug ", "21: add to mug", "22: fixate cupboard", "23: close", "24: fixate spoon",
                          "25: take", "26: fixate mug", "27: stir", "28: fixate table", "29: put down",
                          "30: fixate fridge", "31: open", "32: fixate cream", "33: take", "34: fixate mug",
                          "35: add to mug", "36: fixate fridge", "37: put down", "38: close", "39: fixate spoon",
                          "40: take", "41: fixate mug", "42: stir", "43: fixate table", "44: put down",
                          "45: fixatemug", "46: take", "47: sip", "48: fixate table", "49: put down", "50: good coffee!"])



    plot = sns.barplot(data=errors, orient="h", estimator=np.mean, ci="sd", color='lightblue', errwidth=0.4) #x="Sequence step", y="Error %",
    plot.set_xlabel('Error rate (%)', fontsize=10)
    plot.set_ylabel('Sequence steps', fontsize=10)
    plot.tick_params(labelsize=5)
    plt.tight_layout()


    fig = plot.get_figure()
    fig.savefig(fileto)
    #g = sns.catplot(data=errors, x="day", y="total_bill", kind="violin", inner=None)
    #sns.swarmplot(data=errors, x="day", y="total_bill", color="k", size=3, ax=g.ax)

    #violin_parts = plt.violinplot(errors, widths=.8, showmeans=True, vert=False)
    #plt.tight_layout()
    #plt.savefig(fileto, format='svg')
    #plt.clf()

import matplotlib
def control_robustness_plot(fileto="robustness.svg"):
    avgs = [[171.14, 171.08, 170.96, 166.54, 160.62],
    [77.68, 78.92, 79.7, 81.72, 75.4],
    [26.34, 29.12, 30.82, 30.88, 30.52],
    [9.8, 10.82, 10.6, 12.22, 12.3],
    [3.98, 4.52, 4.98, 5.56, 5.82]]


    ci95high = [[173.6845329, 173.6133859, 174.0716106, 169.5290811, 163.9535618],
    [80.26465809, 81.75490706, 82.7905033, 84.74625468, 78.49463401],
    [27.93593937, 30.87908066, 32.74981585, 32.7560683, 32.63292482],
    [10.74519382, 11.81396788, 11.59944187, 13.43023441, 13.53712101],
    [4.610248366, 5.235382057, 5.658121889, 6.178820266, 6.614942937]]


    ci95low = [[168.5954671, 168.5466141, 167.8483894, 163.5509189, 157.2864382],
    [75.09534191, 76.08509294, 76.6094967, 78.69374532, 72.30536599],
    [24.74406063, 27.36091934, 28.89018415, 29.0039317, 28.40707518],
    [8.854806184, 9.826032116, 9.600558128, 11.00976559, 11.06287899],
    [3.349751634, 3.804617943, 4.301878111, 4.941179734, 5.025057063]]

    for seq in [avgs, ci95high, ci95low]:
        for i, lvl1 in enumerate(seq):
            for j, lvl2 in enumerate(lvl1):
                seq[i][j]= 100*seq[i][j]/210

    fig, ax = plt.subplots()
    for i in range(0, 5):
        ax.plot([1.0, 1.25, 1.50, 1.75, 2.0], avgs[i], marker='+', label=str(i+1))
        ax.fill_between([1.0, 1.25, 1.50, 1.75, 2.0], ci95low[i], ci95high[i], color='b', alpha=.1)
    plt.xticks(ticks=[1.0, 1.25, 1.50, 1.75, 2.0])
    plt.yscale("log")

    legend = ax.legend(loc='center left', bbox_to_anchor=(1,0.5), shadow=False)#, fontsize='x-large')
    legend.set_title("Noise magnitude")
    ax.yaxis.grid(which='major', color='gray', linestyle='dashed', alpha=0.5)
    # Show the minor grid as well. Style it in very light gray as a thin,
    # dotted line.
    ax.yaxis.grid(which='minor', color='gray', linestyle='dashed', alpha=0.2)
    ax.set_yticks(ticks=[1, 5,  10, 50, 100])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlabel('Goal multiplier')
    plt.ylabel('Correct sequences (%)')
    plt.tight_layout()
    plt.savefig(fileto, format='svg')
    plt.clf()