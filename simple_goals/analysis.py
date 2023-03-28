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

def bargraph_with_without_goalunits(filename):
    # Updated info based on 20 networks of each, 1 run each.
    # goals
    means_action_errors_goals = (5.27, 34.25, 60.03, 73.98, 80.65)
    means_subseq_errors_goals = (8.47, 19.24, 19.74, 16.97, 15.19)

    # no goals
    means_action_errors_nogoals = (6.70, 37.78, 61.76, 74.92, 81.76)
    means_subseq_errors_nogoals = (16.60, 23.06, 22.02, 17.71, 14.63)

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