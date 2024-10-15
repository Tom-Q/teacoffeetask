# Entry point of the program. Mostly loads scripts from scripts, which themselves rely on more serious code.
import utils
import sys
from neural import neuralnet as nn, optimizers, layers
import analysis
import goalenv
from goalenv import environment as env, goalenv2020, scripts
import tensorflow as tf
import scipy
import numpy as np

from pnas import pnas2018, pnashierarchy
import rdm

utils.initialize_random_seeds(1212455)

#scripts.basic_analysis(range(20), "bigmodel1_1goals_relu_adam_nonoise", True)
#scripts.basic_analysis(range(20), "bigmodel1_0goals_relu_adam_nonoise", False)
#scripts.plot_pca()
#sys.exit()
"""
#rdm_sum_100_units_goals = scripts.basic_analysis([0, 4, 5, 6, 7], "bigmodel1_1goals_relu_adam_nonoise_100_units", True)
#rdm_sum_100_units_no_goals = scripts.basic_analysis([0, 1, 2, 5, 6], "bigmodel1_0goals_relu_adam_nonoise_100_units", False)
#rdm_sum_goals = scripts.basic_analysis(range(5), "bigmodel1_1goals_relu_adam_nonoise", True)
# rdm_sum_no_goals = scripts.basic_analysis(range(7), "bigmodel1_0goals_relu_adam_nonoise", False)
#print("bigmodel1_1goals_relu_adam_nonoise_100_units")
#rdm_sum_100_units_goals = scripts.basic_analysis([0, 4, 5, 6, 7], "bigmodel1_1goals_relu_adam_nonoise_100_units", True)
#print("bigmodel1_0goals_relu_adam_nonoise_100_units")
rdm_sum_100_units_no_goals = scripts.basic_analysis([0, 1, 2, 4, 6, 9, 10], "bigmodel1_0goals_relu_adam_nonoise_100_units", False) # all those are correct
rdm_sum_no_goals = scripts.basic_analysis(range(7), "bigmodel1_0goals_relu_adam_nonoise", False)
"""
"""
print("bigmodel1_1goals_relu_adam_nonoise")
rdm_sum_goals = scripts.basic_analysis(range(10), "bigmodel1_1goals_relu_adam_nonoise", True) # all correct
print("bigmodel1_0goals_relu_adam_nonoise")
rdm_sum_no_goals = scripts.basic_analysis(range(10), "bigmodel1_0goals_relu_adam_nonoise", False) #

# Save each RDM png
"""
"""
rdm_list = [#("100 units goals", rdm_sum_100_units_goals),
            ("100 units no goals", rdm_sum_100_units_no_goals),
            #("goals", rdm_sum_goals),
            ("no goals", rdm_sum_no_goals)]

# Now correlate those rdms
for i in rdm_list:
    for j in rdm_list:
        print(i[0] + " vs "+ j[0] + "(1. pearson, 2. spearman)")
        print(rdm.compare_matrices_pearson(i[1].matrix, j[1].matrix))
        print(rdm.compare_matrices_spearman(i[1].matrix, j[1].matrix))

for i in rdm_list:
    i[1].save(i[0])
sys.exit()

"""
#for i in range(20):
#    model = pnashierarchy.train_with_goals()
#    utils.save_object("model_with_goals100units_ctt", model)


goalenv.scripts.train_networks_100_units(10)
sys.exit()
#pnas2018.make_rdm_multiple("model_with_goals100units_ctt", 10, with_goals=True, rdm_type=rdm.SPEARMAN)
pnashierarchy.make_rdm_multiple_goals_actions("model_with_goals", 10)
sys.exit(0)
#sys.exit(0)
# Leave-one-out-analysis:
# Still have to train 11 networks (no goals)
#for i in range(2):
goalenv.scripts.train_networks_100_units(6)
#goalenv.scripts.leave_one_out_analysis()


sys.exit(0)
"""
from pnas import pnas2018
l1reg = [0., 0.0001]
l2reg = [0, 0.0001]
algo = [optimizers.SGD, optimizers.RMSPROP, optimizers.ADAM]
size_hidden = [5, 15, 100]
learning_rate = [0.05, 0.1]
loss_type = ["cross_entropy", "mse"]

for l1 in l1reg:
    for l2 in l2reg:
        for alg in algo:
            for size in size_hidden:
                for lr in learning_rate:
                    for loss in ["cross_entropy"]:
                        model, perf = pnas2018.train(iterations=500, l1reg=l1, l2reg=l2, algorithm=alg,
                                               size_hidden=size, learning_rate=lr, loss_type=loss)
                        #_, accuracy_totals = pnas2018.accuracy_test(model)
                        model_desc = "l1{0}_l2{1}_alg{2}_hl{3}_lr{4}_loss{5}_perf{6}".format(str(l1),
                                                                                             str(l2),
                                                                                             alg,
                                                                                             str(size),
                                                                                             str(lr),
                                                                                             loss,
                                                                                             str(perf))
                        utils.save_object(model_desc, model)
                        rdm = pnas2018.make_rdm_multiple(model_desc, num_networks=1, save_name=model_desc)
                        #sys.exit(0)

from reward_task import scripts, reward, rewardtask
scripts.performance_test()
sys.exit()
"""

from goalenv import scripts as gevscripts
#gevscripts.graph_error_rate_per_step_example()
#gevscripts.bargraph_with_without_goalunits("error_bars.png")
#for net in range(10):
#gevscripts.rdm_distances_analysis()
#sys.exit()
#gevscripts.analyse_rdm_distances_output()
#gevscripts.rdm_distances_analysis()
#gevscripts.figure_steps_to_error("myfig.svg")

#from goalenv import scripts

#scripts.basic_analysis()


from atariACC import tests

tests.a2c_test()

sys.exit(0)

from modularACC import predictive
import rdm

for i in range(2):
    mod, _ = predictive.train_new_task(ultra=True, size_acc=50, iterations=20000, learning_rate_acc=0.1)
    utils.save_object("pred_acc_ultra2", mod)
rdm_acc, rdm_context = predictive.make_rdm_multiple_new_task("pred_acc_ultra2", num_networks=5, save_files=True, skips=[], rdm_type=rdm.EUCLIDIAN)
rdm_acc.save("rdm_acc_ultra2")
rdm_context.save("rdm_ctxt_ultra2")


for i in range(2):
    mod, _ = predictive.train_new_task(ultra=False, size_acc=50, iterations=20000, learning_rate_acc=0.1)
    utils.save_object("pred_acc2", mod)
rdm_acc, rdm_context = predictive.make_rdm_multiple_new_task("pred_acc2", num_networks=5, save_files=True, skips=[], rdm_type=rdm.EUCLIDIAN)
rdm_acc.save("rdm_acc2")
rdm_context.save("rdm_ctxt2")


"""
from modularACC import criticACC
import rdm
for i in range(2):
    mod, _ = criticACC.train(size_acc=20, iterations=2000, learning_rate_acc=0.1)
    utils.save_object("critic_acc", mod)
rdm_acc = criticACC.make_rdm_multiple("critic_acc", num_networks=3, save_files=True, skips=[], rdm_type=rdm.EUCLIDIAN)
rdm_acc.save("rdm_critic_acc")
exit()
#utils.save_object("rdm_acc", rdm_acc)
rdm = utils.load_object("rdm_acc")
rdm.save("rdm_critic_acc_log")
rdm.delete_entry(6)
rdm.delete_entry(0)
rdm.save("steps_1to5")
rdm.delete_entry(5)
rdm.delete_entry(0)
rdm.save("steps_1to4")
rdm.delete_entry(4)
rdm.delete_entry(0)
rdm.save("steps_1to3")
rdm.delete_entry(3)
rdm.delete_entry(0)
rdm.save("steps_1to2")
#rdm_context.save("rdm_ctxt")
"""

"""
# peaks subproject
# PHASE 1.
import csv
import scipy.spatial.kdtree
import geopy

class Summit(object):
    def __init__(self, elevation, latitude, longitude):
        self.elevation = elevation
        self.latitude = latitude
        self.longitude = longitude
        self.hash = hash((self.latitude, self.longitude))

# Load the list of summits, constructing summit structs
summit_list = []
with open('all-peaks-sorted-p100.txt', 'r') as file:
    reader = csv.reader(file)
    for i, row in enumerate(reader):
        summit_list.append(Summit(row[2], row[0], row[1]))
        if i % 100000 == 0:
            print(i)
        if i > 1000000:  # for now just use 1,000,000 summits for testing purposes
            break

# Sort the list by elevation
sorted(summit_list, key=lambda x:x.elevation, reverse=True)

# Make a dict to have a O(1) access to the summits based on coordinates
summit_dict = {}
for summit in summit_list:
    summit_dict[summit.hash] = summit

# Make a ball tree
from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd

cities = pd.DataFrame(data={
    'name': [...],
    'lat': [...],
    'lon': [...]
})

query_lats = [...]
query_lons = [...]

bt = BallTree(np.deg2rad(cities[['lat', 'lon']].values), metric='haversine')
distances, indices = bt.query(np.deg2rad(np.c_[query_lats, query_lons]))

nearest_cities = cities['name'].iloc[indices]


tree = scipy.spatial.kdtree.KDTree(data)

R = 6378137 # earth radius in meters.
# Iterate over each summit, looking for neighbors.
for summit in summit_list:
    # evaluate twice the horizon distance
    horizon_distance = np.sqrt(2 * R * summit.elevation + summit.elevation**2)
    doubled = horizon_distance * 2

    # look for every mountain within twice horizon distance
    candidates = tree.query_ball_point((summit.latitude, summit.longitude), doubled)
    for candidate in candidates:

print("done")
sys.exit(0)

# Sort it by elevation
sorted(summits, elevation)
compute theoretical max visual range
summits2 = copy(summits)
sorted(summits2, latitude + longitude) # meaning I need to box them by longitude/latitude.
for summit in summits:
    # find all summits which "see each other"
    # 1. Find all summits within latitude + longitude range
    for candidate_summits:
        # 1. Check whether candidate < summit
        # 2. Check whether they theoretically see each other
        # 3. Compute the angle from summit 2 to summit 1
        # if angle < x delete summit from all lists (not a top summit = relevant)
# Result = list of summits that can't see anything higher than them by looking up

# PHASE 2.
for summit in summits:
    visibles = # find all summits which theoretically see this summit.
    sorted(visibles, elevation)
    for visible:

"""