# Entry point of the program. Mostly loads scripts from scripts, which themselves rely on more serious code.
import teacoffeeenv as tce
import scripts
import neuralnet as nn
import utils
import pnas2018
import sys

#scripts.train_supervised(nn.NeuralNet(), 10000)

pnas2018.make_rdm_multiple('lopsided_75_25', 20, with_goals=False, title="lopsided 2020-09-10")  # reuse that 'pnas_goals_ce_5000' for probas with goals
#pnas2018.make_rdm_and_mds_pnas('pnas2')

sys.exit("end of program")

"""
for i in range(100):
    model = pnas2018.train_pnas(noise=0.0, iterations=5000)
    pnas2018.accuracy_test_pnas(model)
    print(i)
    utils.save_object("pnas_regularization001_5000", model)

"""
for i in range(20):
    model = pnas2018.train_pnas(noise=0., iterations=5000, lopsided=True)
    pnas2018.accuracy_test_pnas(model)
    print(i)
    utils.save_object("lopsided_75_25", model) # rerun. Rename this to reg001
