from itertools import combinations
import numpy as np

def fitness_fn_mean_ens(genome, prediction_map, actuals, ensemble_size):
    
    def get_constituent_ensembles(genome, ensembles):
        return [ensemble for ensemble in ensembles if genome in ensemble]
    
    def average_activations(ensemble):
        a = [prediction_map[genome] for genome in ensemble]
        return np.mean(a, axis=0)
    
    ensembles = combinations(prediction_map, r = ensemble_size)
    constituent_ensembles = get_constituent_ensembles(genome, ensembles)
    
    ensemble_accuracies = []
    
    for ens in constituent_ensembles:
        ensemble_average_activations = average_activations(ens)
        ensemble_predictions = [np.argmax(a) for a in ensemble_average_activations]
        ensemble_accuracies.append(np.mean(([pred == actual for pred, actual in zip(ensemble_predictions, actuals)])))
    
    genome.fitness = np.mean(ensemble_accuracies)
    
    return genome