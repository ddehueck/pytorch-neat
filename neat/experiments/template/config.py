import torch
import torch.nn as nn
import numpy as np
from neat.phenotype.feed_forward import FeedForwardNet

from utils import create_prediction_map, random_ensemble_generator_for_static_genome


class TemplateConfig:

    #TODO define __repr__(self) to return experiment parameters for SA

    def __init__(self, **kwargs):

        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for k, v in kwargs.items(): 
            setattr(self, k, v)


    def eval_genomes(self, genomes):

        #GET RID OF THIS | REPLACE WITH ALG SELECTED BY KWARG
        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            return np.exp(x)/np.sum(np.exp(x),axis=0)

        def cross_entropy(y,y_pred):
            loss=-np.sum(y*np.log(y_pred))
            return loss/float(y_pred.shape[0])
        
        dataset = [] #TODO get [tensors] self.DATASET
        y = [] #TDOD get [actuals]

        activations_map = create_prediction_map(genomes, dataset, self)

        for genome in genomes:
            constituent_ensemble_losses = []
            #Iterate through a sample of all possible combinations of candidate genomes to ensemble for a given size k
            for sample_ensemble in random_ensemble_generator_for_static_genome(genome, genomes, k = self.GENERATIONAL_ENSEMBLE_SIZE, limit = self.CANDIDATE_LIMIT): #Returns limit length iterable of array of size k of dict {genome:activations} 
                #Append given genome activations to list
                ensemble_activations = [activations_map[genome]]
                #Append candidate genome activations to list
                for candidate in sample_ensemble:
                    ensemble_activations.append(activations_map[candidate])
                average_ensemble_activations = np.mean(ensemble_activations, axis = 0)
                ensemble_predictions = np.array([softmax(a) for a in average_ensemble_activations]) #TODO Replace with function specified by config kwarg 
                constituent_ensemble_loss = cross_entropy(y,ensemble_predictions) 
                constituent_ensemble_losses.append(constituent_ensemble_loss)
            #set the genome fitness as the average loss of the candidate ensembles TODO use kwarg switching for fitness_fn
            genome.fitness = np.mean(constituent_ensemble_losses)
        
        population_fitness = np.mean([genome.fitness for genome in genomes])

        return population_fitness
