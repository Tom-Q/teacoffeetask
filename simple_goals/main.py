# Entry point of the program. Mostly loads scripts from scripts, which themselves rely on more serious code.
import teacoffeeenv as tce
import scripts
import neuralnet as nn
import utils
import pnas2018

#scripts.train_supervised(nn.NeuralNet(), 10000)
pnas2018.make_rdm_multiple('pnasrdm', 100, with_goals=False)
#pnas2018.make_rdm_and_mds_pnas('pnas2')


"""
for i in range(7):
    model = pnas2018.train_pnas()
    pnas2018.accuracy_test_pnas(model)
    utils.save_object("pnasrdmnogoals", model)

"""