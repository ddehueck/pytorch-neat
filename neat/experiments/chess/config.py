import torch
import logging
import torch.nn as nn
import pickle
import numpy as np
from torch import autograd
from neat.phenotype.feed_forward import FeedForwardNet

logger = logging.getLogger(__name__)

class ChessConfig:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # DEVICE = "cuda:0"
    VERBOSE = True

    NUM_INPUTS = 8
    NUM_OUTPUTS = 1
    USE_BIAS = True

    ACTIVATION = 'tanh'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 1970

    POPULATION_SIZE = 100
    NUMBER_OF_GENERATIONS = 150
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.0
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.30
    print("loading chess data")
    print(DEVICE)
    with open("./data/KQK/indices", "rb") as f: 
        tensors = pickle.load(f)
        # print(tensors)
    inputs_list = []
    outputs_list = []
    np.random.shuffle(tensors)
    tensors = tensors[:2000]
    for tensor in tensors:
        tensor_in = np.array(tensor[0])
        # condensed_input = np.argmax(tensor_in.reshape(4, 64), axis=1)
        inputs_list.append(tensor_in)
        outputs_list.append(tensor[1])

    inputs = list(map(lambda s: autograd.Variable(torch.Tensor([s])), inputs_list))
    
    targets = list(map(lambda s: autograd.Variable(torch.Tensor([s])), outputs_list))

    def fitness_fn(self, genome):
        fitness = 2000  # Max fitness for XOR

        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)
        criterion = nn.MSELoss()
        num_inputs = len(self.inputs)
        for input, target in zip(self.inputs, self.targets):  # 4 training examples
            input, target = input.to(self.DEVICE), target.to(self.DEVICE)

            pred = phenotype(input)
            # print(pred)
            # print(target)
            loss = (float(pred) - float(target)) ** 2
            loss = float(loss)
            # loss = criterion(pred, target)
            # logger.info("Loss: {}".format(loss))
            fitness -= loss
            # logger.info("Fitness: {}".format(fitness))
        # fitness = fitness / num_inputs
        # logger.info("Fitness: {}".format(fitness))
        return fitness

    def get_preds_and_labels(self, genome):
        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)

        predictions = []
        labels = []
        for input, target in zip(self.inputs, self.targets):  # 4 training examples
            input, target = input.to(self.DEVICE), target.to(self.DEVICE)

            predictions.append(float(phenotype(input)))
            labels.append(float(target))

        return predictions, labels
