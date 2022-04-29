import numpy as np
# Identical to PNAS except without the first step.

all_inputs = ["coffee_stim", "tea_stim", "coffee", "milk", "water", "stir", "tea", "sugar", "serve_coffee", "serve_tea"]
all_outputs = ["coffee", "milk", "water", "stir", "sugar", "tea", "serve_coffee", "serve_tea"]
seq1 = ["coffee_stim", "coffee", "water", "stir", "milk", "stir", "serve_coffee"]
seq2 = ["coffee_stim", "coffee", "milk", "stir", "water", "stir", "serve_coffee"]
seq3 = ["tea_stim", "tea", "water", "stir", "sugar", "stir", "serve_tea"]
seq4 = ["tea_stim", "tea", "sugar", "stir", "water", "stir", "serve_tea"]
seqs = [seq1, seq2, seq3, seq4]

# Special goals
length = len(seq1) -1
goals = [[[1., 0., 0., 0.]]*length, [[0., 1., 0., 0.]]*length, [[0., 0., 1., 0.]]*length, [[0., 0, 0, 1]]*length]
goals = [np.asarray(goal, dtype=np.float32).reshape((-1, 1, 4)) for goal in goals]
#goals = [[[0., 1.]]*6, [[0., 1.]]*6, [[1., 0]]*6, [[1, 0]]*6]
#goals = [np.asarray(goal, dtype=np.float32).reshape((-1, 1, 2)) for goal in goals]
sequence_probabilities=[0.25, 0.25, 0.25, 0.25]
