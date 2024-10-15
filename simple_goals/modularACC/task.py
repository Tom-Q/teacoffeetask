all_inputs = ["tea leaves", "coffee grounds", "pepper", "spoon", "fork", "knife", "orange juice", "coffee cup",
              "tea cup", "boiling water", "sugar", "milk", "happy face", "sad face",
              "tea intent", "coffee intent", "water first intent", "milk first intent", "sugar first intent"]
all_outputs = ["tea leaves", "coffee grounds", "pepper", "spoon", "fork", "knife", "orange juice", "coffee cup",
              "tea cup", "boiling water", "sugar", "milk"]

input_seq = [["tea leaves", "coffee grounds", "pepper"],
              ["boiling water", "sugar", "milk"],
              ["spoon", "fork", "knife"],
              ["boiling water", "sugar", "milk"],
              ["spoon", "fork", "knife"],
              ["orange juice", "coffee cup", "tea cup"]]

seq1 = coffee_water_first = ["coffee grounds", "boiling water", "spoon", "milk", "spoon", "coffee cup"]
seq2 = coffee_water_second = ["coffee grounds", "milk", "spoon", "boiling water", "spoon", "coffee cup"]
seq3 = tea_water_first = ["tea leaves", "boiling water", "spoon", "sugar", "spoon", "tea cup"]
seq4 = tea_water_second = ["tea leaves", "sugar", "spoon", "boiling water", "spoon", "tea cup"]
seqs = [seq1, seq2, seq3, seq4]
#sequence_probabilities=[0.25, 0.25, 0.25, 0.25]


# Optimal performance:
# [0.5, 0.5, 1, 1, 1, 1]
# OR
# [1, 0.5, 1, 1, 1, 1]