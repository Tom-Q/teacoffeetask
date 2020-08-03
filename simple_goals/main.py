# Entry point of the program. Mostly loads scripts from scripts, which themselves rely on more serious code.
import teacoffeeenv as tce
import scripts
import neuralnet as nn

scripts.train_supervised(nn.NeuralNet(), 10000)
#scripts.make_rdm_and_mds('noise_test2')