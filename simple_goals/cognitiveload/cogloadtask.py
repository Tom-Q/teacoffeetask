# basic neural net that subtracts
import utils

# Whether to train on ARI, BEV, or BOTH
ONLY_ARI = "only_ari"
ONLY_BEV = "only_bev"
BOTH = "both"
START_BEV = "start_bev"
START_ARI = "start_ari"
STARTS = [START_BEV, START_ARI]

output_symbols = ['0', '1', '2', '4', '6', '+', '-', '=',
           "tea", "coffee", "water", "stir", "sugar", "cream", "serve_tea", "serve_coffee"]

input_symbols = ['1', '2', '3', '+', '-', '=',
                 "choose", "tea", "coffee", "water", "stir", "sugar", "cream"]

symbols = list(set(output_symbols + input_symbols))

goal_symbols = ["math", "beverage"]

beverage_seqs = [
    ["choose", "coffee", "water", "stir", "cream", "stir", "serve_coffee"],
    ["choose", "coffee", "cream", "stir", "water", "stir", "serve_coffee"],
    ["choose", "tea", "water", "stir", "sugar", "stir", "serve_tea"],
    ["choose", "tea", "sugar", "stir", "water", "stir", "serve_tea"]
]

arithmetic_seqs = []
"""
plusplusseqs = [
[3,4,5,12],[4,3,6,13],[5,4,3,12],[6,3,4,13],
[7,4,2,13],[8,5,4,17],[9,6,3,18],[7,4,3,14],
[3,5,7,15],[4,5,8,17],[5,3,4,12],[6,5,3,14],
[7,5,6,18],[8,4,7,19],[9,2,5,16],[6,8,3,17],
[6,5,7,18]]
for seq in plusplusseqs:
    for idx, elem in enumerate(seq):
        seq[idx]=str(elem)
    seq.insert(1, '+')
    seq.insert(3, '+')
    seq.insert(5, '=')
    arithmetic_seqs.append(seq)
plusminusseqs = [
    [6, 9, 8, 7], [4,2,9,-3], [5,3,9,-1],
    [6,5,7,4], [4,7,3,8], [7,5,6,6],
    [8,4,5,7], [5,2,8,-1], [7,5,6,6],
    [4,3,9,-2], [5,2,8,-1], [6,3,4,5],
    [4,7,3,8], [2,3,8,-3], [5,6,3,8],
    [2,3,8,-3], [9,4,7,6],
    ]
for seq in plusminusseqs:
    for idx, elem in enumerate(seq):
        seq[idx]=str(elem)
    seq.insert(1, '+')
    seq.insert(3, '-')
    seq.insert(5, '=')
    arithmetic_seqs.append(seq)
minusplusseqs = [
    [9,2,5,12],    [6,7,8,7],    [3,8,4,-1],
    [4,3,7,8],    [8,4,5,9],    [7,2,4,9],
    [8,3,6,11],    [9,4,3,8],    [2,5,7,4],
    [3,9,4,-2],    [4,8,3,-1],    [5,3,6,8],
    [2,9,4,-3],    [3,8,2,-3],    [4,9,3,-2],
    [5,7,9,7],    [9,5,3,7]
]
for seq in minusplusseqs:
    for idx, elem in enumerate(seq):
        seq[idx]=str(elem)
    seq.insert(1, '-')
    seq.insert(3, '+')
    seq.insert(5, '=')
    arithmetic_seqs.append(seq)
minusminusseqs = [
    [9,6,2,1],    [4,9,4,-9],    [6,8,3,-5],    [8,5,4,-1],
    [7,4,5,-2],    [3,4,6,-7],    [4,5,6,-7],    [5,8,3,-6],
    [5,2,8,-5],    [4,3,5,-4],    [7,3,5,-1],    [2,6,3,-7],
    [6,8,5,-7],    [8,2,5,1],    [7,3,6,-2],    [7,3,8,-4],
    [9,5,2,2]
    ]
for seq in minusminusseqs:
    for idx, elem in enumerate(seq):
        seq[idx]=str(elem)
    seq.insert(1, '-')
    seq.insert(3, '-')
    seq.insert(5, '=')
    arithmetic_seqs.append(seq)
"""

"""
arithmetic_seqs_easy = [['1', '+', '1', '-', '2', '=', '0'],
                        ['2', '+', '1', '-', '2', '=', '1'],
                        ['1', '+', '2', '-', '1', '=', '2'],
                        ['2', '+', '2', '-', '1', '=', '3']]

label_seqs_ari = [['1->+', '+->1', '1->-', '-->2', '2->=', '=->0'],
                  ['2->+', '+->1', '1->-', '-->2', '2->=', '=->1'],
                  ['1->+', '+->2', '2->-', '-->1', '1->=', '=->2'],
                  ['2->+', '+->2', '2->-', '-->1', '1->=', '=->3']
]
"""
arithmetic_seqs_easy = [['3', '+', '2', '+', '1', '=', '6'],
                        ['3', '+', '2', '-', '1', '=', '4'],
                        ['3', '-', '2', '+', '1', '=', '2'],
                        ['3', '-', '2', '-', '1', '=', '0']]

label_seqs_ari = [['3->+', '+->2', '2->+', '+->1', '1->=', '=->6'],
                  ['3->+', '+->2', '2->-', '-->1', '1->=', '=->4'],
                  ['3->-', '-->2', '2->+', '+->1', '1->=', '=->2'],
                  ['3->-', '-->2', '2->-', '-->1', '1->=', '=->0']
]
# Model 2: we're training only one network. This network is trained to perform all 3 tasks.
# To achieve this, when training for bev or ari we train on the same time-steps (=with a blank timestep in between)
label_seqs_ari_blanks = [['num1', '0', '+', '0', 'num2', '0', '+', '0', 'num3', '0', '='],
                        ['num1', '0', '+', '0', 'num2', '0', '-', '0', 'num3', '0', '='],
                        ['num1', '0', '-', '0', 'num2', '0', '+', '0', 'num3', '0', '='],
                        ['num1', '0', '-', '0', 'num2', '0', '-', '0', 'num3', '0', '=']]
label_seqs_bev_blanks = [
    ["coffee", '0', "water", '0', "stir", '0', "cream", '0', "stir", '0', "serve_coffee"],
    ["coffee", '0', "cream", '0', "stir", '0', "water", '0', "stir", '0', "serve_coffee"],
    ["tea", '0', "water", '0', "stir", '0', "sugar", '0', "stir", '0', "serve_tea"],
    ["tea", '0', "sugar", '0', "stir", '0', "water", '0', "stir", '0', "serve_tea"]
]

label_seqs_ari_noblanks = label_seqs_ari

label_seqs_bev_noblanks = [
    ["choose->coffee", "coffee->water", "water->stir", "stir->cream", "cream->stir", "stir->serve_coffee"],
    ["choose->coffee", "coffee->cream", "cream->stir", "stir->water", "water->stir", "stir->serve_coffee"],
    ["choose->tea", "tea->water", "water->stir", "stir->sugar", "sugar->stir", "stir->serve_tea"],
    ["choose->tea", "tea->sugar", "sugar->stir", "stir->water", "water->stir", "stir->serve_tea"]
]

label_seqs_bev = []
for seq in beverage_seqs:
    label_seqs_bev.append(seq[1:])

class Target(object):
    def __init__(self, action, goal1=None):
        self.action_one_hot = action
        if isinstance(goal1, str):
            goal1 = utils.str_to_onehot(goal1, goal_symbols)
        self.goal1_one_hot = goal1
        self.goal2_one_hot = None

# 2 goals option
goal_target_bev = ['beverage']*6
goal_target_ari = ['math']*6


RDM_MODE_AVERAGE_ACTIVATIONS = "rdm_mode_average_activations"
RDM_MODE_AVERAGE_DISTANCES = "rdm_mode_average_distances"