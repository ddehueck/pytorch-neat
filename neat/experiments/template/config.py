import torch
import torch.nn as nn
import numpy as np
from neat.phenotype.feed_forward import FeedForwardNet

from utils import create_prediction_map, random_ensemble_generator_for_static_genome


class TemplateConfig:

    #TODO define __repr__(self) to return experiment parameters for SA

    def __init__(self, **kwargs):

        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.VERBOSE = True

        #DATASET KWARGS
        self.DATASET = kwargs['DATASET']

        self.NUM_INPUTS = kwargs['NUM_INPUTS']
        self.NUM_OUTPUTS = kwargs['NUM_OUTPUTS']
        self.USE_BIAS = kwargs['USE_BIAS']
        
        #ENSEMBLE KWARGS
        self.GENERATIONAL_ENSEMBLE_SIZE = kwargs['ENSEMBLE_SIZE']

        self.ACTIVATION = kwargs['ACTIVATION']
        self.SCALE_ACTIVATION = kwargs['SCALE_ACTIVATION']

        self.FITNESS_THRESHOLD = kwargs['FITNESS_THRESHOLD']

        self.POPULATION_SIZE = kwargs['POPULATION_SIZE']
        self.NUMBER_OF_GENERATIONS = kwargs['NUMBER_OF_GENERATIONS']
        self.SPECIATION_THRESHOLD = kwargs['POPULATION_SIZE']

        self.CONNECTION_MUTATION_RATE = kwargs['CONNECTION_MUTATION_RATE']
        self.CONNECTION_PERTURBATION_RATE = kwargs['CONNECTION_PERTURBATION_RATE']
        self.ADD_NODE_MUTATION_RATE = kwargs['ADD_NODE_MUTATION_RATE']
        self.ADD_CONNECTION_MUTATION_RATE = kwargs['ADD_CONNECTION_MUTATION_RATE']

        self.CROSSOVER_REENABLE_CONNECTION_GENE_RATE = kwargs['CROSSOVER_REENABLE_CONNECTION_GENE_RATE']

        # Top percentage of species to be saved before mating
        self.PERCENTAGE_TO_SAVE = kwargs['PERCENTAGE_TO_SAVE']


    def fitness_fn(self, genome):

        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)
        fitness = np.inf

        for input, target in zip(self.inputs, self.targets):  # 4 training examples
            input, target = input.to(self.DEVICE), target.to(self.DEVICE)

            print(input.shape)
            pred = phenotype(input)
            # Run softmax on the output
            pred = nn.functional.softmax(pred, dim=0)
            # Get the index of the max log-probability
            # pred = pred.argmax(dim=0, keepdim=True)
            # Compute the loss
            print("Pred:", pred.shape)
            print("Target:", target.shape)

            pred = pred.reshape(10)
            # Convert pred to long
            pred = pred

            target = target.reshape(10)
            # Convert target to long
            target = target


            # Compute the loss
            loss = nn.functional.mse_loss(pred, target)

            # Compute the fitness
            fitness -= loss.item()
            # loss = criterion(pred, target)

        return fitness

    def eval_genomes(self, genomes):
        
        dataset = [] #TODO get [tensors] self.DATASET

        activations_map = create_prediction_map(genomes, dataset, self)

        for genome in genomes:
            sample_ensembles = random_ensemble_generator_for_static_genome(genome, genomes, k = self.GENERATIONAL_ENSEMBLE_SIZE) #Returns dict {genome:activations}

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
