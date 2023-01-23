import torch
import torch.nn as nn
from torch import autograd
from neat.phenotype.feed_forward import FeedForwardNet
#from torchvision import datasets
from tqdm import tqdm

from neat.utils import create_prediction_map, random_ensemble_generator_for_static_genome, speciate
import neat.analysis.wrapper as wrapper

import numpy as np

# import wandb

class UCIConfig:
    

    def __init__(self, **kwargs):

        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for k, v in kwargs.items(): 
            setattr(self, k, v)

        if self.USE_FITNESS_COEFFICIENT: #type: ignore

            increment = (self.FINAL_FITNESS_COEFFICIENT - self.INITIAL_FITNESS_COEFFICIENT)/self.NUMBER_OF_GENERATIONS  # type: ignore

            ensemble_coefficients = np.arange(self.INITIAL_FITNESS_COEFFICIENT, self.FINAL_FITNESS_COEFFICIENT, increment)  # type: ignore
            genome_coefficients = ensemble_coefficients[::-1]
            self.genome_coefficients = iter(genome_coefficients)
            self.ensemble_coefficients = iter(ensemble_coefficients)
        
        else:
            genome_coefficients = np.ones(self.NUMBER_OF_GENERATIONS) #type: ignore
            ensemble_coefficients = np.zeros(self.NUMBER_OF_GENERATIONS) #type: ignore
            self.genome_coefficients = iter(genome_coefficients)
            self.ensemble_coefficients = iter(ensemble_coefficients)


    def __call__(self):
        return self

    def create_activation_map(self, genomes, X):
        genomes_to_results = {}
        for genome in tqdm(genomes):
            results = []
            phenotype = FeedForwardNet(genome, self)
            phenotype.to(self.DEVICE)
            for input in X:
                #Adds batch dimension
                input = torch.unsqueeze(input, 0)
                input.to(self.DEVICE)
                prediction = phenotype(input).to('cpu')
                results.append(prediction)
            genomes_to_results[genome] = torch.squeeze(torch.stack(results))
        return genomes_to_results
    
    def constituent_ensemble_evaluation(self, ensemble_activations):
        
        softmax = nn.Softmax(dim=1)
        CE_loss = nn.CrossEntropyLoss()

        soft_activations = torch.sum(torch.stack(ensemble_activations, dim = 0), dim = 0)
        
        constituent_ensemble_loss = CE_loss(softmax(soft_activations), self.TEST_TARGET.to(torch.float32)).item()

        ensemble_fitness = np.exp(-1 * constituent_ensemble_loss)

        self.wandb.log({"constituent_ensemble_loss": constituent_ensemble_loss})
        self.wandb.log({"constituent_ensemble_fitness": ensemble_fitness})

        return ensemble_fitness

    def eval_genomes(self, genomes):

        activations_map = self.create_activation_map(genomes, self.DATA) #type: ignore

        #print(activations_map.values())
        
        genome_fitness_coefficient = next(self.genome_coefficients)
        ensemble_fitness_coefficient = next(self.ensemble_coefficients)

        #print(f"fitness = {genome_fitness_coefficient} * genome_fitness + {ensemble_fitness_coefficient} * constituent_ensemble_fitness")
        # species = speciate(genomes, self.SPECIATION_THRESHOLD) #type: ignore
        # print(species)

        for genome in tqdm(genomes):
            softmax = nn.Softmax(dim=1)
            genome_prediction = softmax(activations_map[genome])
            CE_loss = nn.CrossEntropyLoss()
            genome_loss = CE_loss(genome_prediction, self.TARGET.to(torch.float32)).item()
            genome_fitness = np.exp(-1 * genome_loss)
            #print(f"genome_loss: {genome_loss} | genome_fitness: {genome_fitness}")
            constituent_ensemble_losses = []
            #Iterate through a sample of all possible combinations of candidate genomes to ensemble for a given size k
            

            sample_ensembles = random_ensemble_generator_for_static_genome(genome, genomes, k = self.GENERATIONAL_ENSEMBLE_SIZE, limit = self.CANDIDATE_LIMIT)  # type: ignore

            for sample_ensemble in sample_ensembles:

                ensemble_activations = [activations_map[genome]]

                #Append candidate genome activations to list
                for candidate in sample_ensemble:
                    ensemble_activations.append(activations_map[candidate])
                
                #TODO: implement Hard voting
                soft_activations = torch.sum(torch.stack(ensemble_activations, dim = 0), dim = 0)
                
                constituent_ensemble_loss = CE_loss(softmax(soft_activations), self.TARGET.to(torch.float32)).item()

                constituent_ensemble_losses.append(constituent_ensemble_loss)

            #set the genome fitness as the average loss of the candidate ensembles TODO use kwarg switching for fitness_fn
            
            ensemble_fitness = np.exp(-1 * np.mean(constituent_ensemble_losses))

            #print(f"ensemble_loss: {np.mean(constituent_ensemble_losses)} | ensemble_fitness: {ensemble_fitness}")
            
            genome.fitness = genome_fitness_coefficient * genome_fitness + ensemble_fitness_coefficient * ensemble_fitness
            print(genome.species)

        df_results = wrapper.run_trial_analysis(self.create_activation_map(genomes, self.TEST_DATA), self.constituent_ensemble_evaluation)
        df_results.to_csv('./df_results.csv')

        # Take the mean for each column
        # df_results = 
        # Convert it to a dictinary
        # df_results = df_results
        # Log the results with wandb
        self.wandb.log(df_results.mean(axis=0).to_dict())
        
        population_fitness = np.mean([genome.fitness for genome in genomes])
        #print("population_fitness: ", population_fitness)
        
        # Log population fitness with wandb 
        self.wandb.log({"population_fitness": population_fitness})
        return population_fitness
