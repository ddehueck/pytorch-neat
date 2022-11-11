import torch
import torch.nn as nn
from torch import autograd
from neat.phenotype.feed_forward import FeedForwardNet
# Import the MNIST dataset from torchvision
from torchvision import datasets

from neat.utils import create_prediction_map, random_ensemble_generator_for_static_genome

import numpy as np


class MNISTConfig:
    

    def __init__(self, **kwargs):

        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for k, v in kwargs.items(): 
            setattr(self, k, v)

        mnist_data = datasets.MNIST(root="./data", train=True, download=True)
        data = mnist_data.data
        data = data.view(data.size(0), -1).float()
        data = data / 255
        self.targets = mnist_data.targets

        # self.test = mnist_data.test_data
        # self.test = self.test.view(self.test.size(0), -1).float()
        # self.test = self.test / 255
        # self.test_labels = mnist_data.test_labels

        # train = train[:10]
        # train_labels = train_labels[:10]
        # test = test[:10]
        # test_labels = test_labels[:10]
        # Split all training examples into a python list
        self.data = list(data)
        self.data = [i.reshape(1, 784) for i in self.data]
        print(len(self.data))
        print(self.data[0].shape)

        self.data = self.data[:10]
        self.targets = self.targets[:10]
        # Print the shape of the train dataset
        #print("Data shape:", self.data)
        # Print the shape of the test dataset
        #print("Test shape:", self.test.shape)
    
        # Print the targets
        self.targets = torch.from_numpy(np.eye(10)[self.targets])
        self.targets = list(self.targets)
        self.targets = [i.reshape(1, 10) for i in self.targets]
        print("Targets:", len(self.targets))
        print("Target shape:", self.targets[0].shape)

    def __call__(self):
        return self


    def eval_genomes(self, genomes):

        #GET RID OF THIS | REPLACE WITH ALG SELECTED BY KWARG
        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            return np.exp(x)/np.sum(np.exp(x),axis=0)

        def cross_entropy(y,y_pred):
            loss=-np.sum(y*np.log(y_pred))
            return loss/float(y_pred.shape[0])
        
        dataset = self.data #TODO get [tensors] self.DATASET
        y = [np.squeeze(np.array(y_)) for y_ in self.targets] #TDOD get [actuals]
        #print("Y:", y)

        activations_map = create_prediction_map(genomes, dataset, self)

        #print("map:" , activations_map)

        for genome in genomes:
            constituent_ensemble_losses = []
            #Iterate through a sample of all possible combinations of candidate genomes to ensemble for a given size k
            sample_ensembles = random_ensemble_generator_for_static_genome(genome, genomes, k = self.GENERATIONAL_ENSEMBLE_SIZE, limit = self.CANDIDATE_LIMIT)
            #print("sample_ENS:", [s for s in sample_ensembles])
            for sample_ensemble in sample_ensembles: #Returns limit length iterable of array of size k of dict {genome:activations} 
                #Append given genome activations to list
                #print("sample_ens: ", sample_ensemble)
                #print("genome_Z", activations_map[genome]

                ensemble_activations = [np.squeeze(activations_map[genome])]
                #Append candidate genome activations to list
                for candidate in sample_ensemble:
                    ensemble_activations.append(np.squeeze(activations_map[candidate]))
                
                #print("ens_Z", ensemble_activations)
                average_ensemble_activations = np.mean(ensemble_activations, axis = 0)
                #print("mean_ens_Z", average_ensemble_activations)
                ensemble_predictions = np.array([softmax(z) for z in average_ensemble_activations]) #TODO Replace with function specified by config kwarg
                #print("ens pred:", ensemble_predictions[0]) 
                constituent_ensemble_loss = cross_entropy(y,ensemble_predictions)
                constituent_ensemble_losses.append(constituent_ensemble_loss)
            #set the genome fitness as the average loss of the candidate ensembles TODO use kwarg switching for fitness_fn
            
            
            genome.fitness = np.exp(-1 * np.mean(constituent_ensemble_losses))
            print(f"{id(genome)} : {genome.fitness}")
        
        population_fitness = np.mean([genome.fitness for genome in genomes])
        print("population_fitness: ", population_fitness)
        #return population_fitness