import numpy as np

all_inputs = ["start", "coffee", "milk", "water", "stir", "tea", "sugar", "serve_coffee", "serve_tea"]
all_outputs = ["coffee", "milk", "water", "stir", "sugar", "tea", "serve_coffee", "serve_tea"]
seq1 = ["start", "coffee", "water", "stir", "milk", "stir", "serve_coffee"]
seq2 = ["start", "coffee", "milk", "stir", "water", "stir", "serve_coffee"]
seq3 = ["start", "tea", "water", "stir", "sugar", "stir", "serve_tea"]
seq4 = ["start", "tea", "sugar", "stir", "water", "stir", "serve_tea"]
seqs = [seq1, seq2, seq3, seq4]

# Special goals
goals = [[[1., 0.]]*6, [[1., 0.]]*6, [[0., 1.]]*6, [[0., 1.]]*6]
goals = [np.asarray(goal, dtype=np.float32).reshape((-1, 1, 2)) for goal in goals]
#goals = [[[0., 1.]]*6, [[0., 1.]]*6, [[1., 0]]*6, [[1, 0]]*6]
#goals = [np.asarray(goal, dtype=np.float32).reshape((-1, 1, 2)) for goal in goals]
sequence_probabilities=[0.25, 0.25, 0.25, 0.25]

# Optimal performance:
# [0.5, 0.5, 1, 1, 1, 1]
# OR
# [1, 0.5, 1, 1, 1, 1]