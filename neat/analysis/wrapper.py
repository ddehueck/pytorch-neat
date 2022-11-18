import pandas as pd

from neat.analysis.algorithms import ALGORITHMS


def run_trial_analysis(final_population_prediction_map, ensemble_evaluator):
    algorithm_results = {
        name: algo(final_population_prediction_map, ensemble_evaluator)
        for name, algo in ALGORITHMS.items()
    }
    return pd.DataFrame(algorithm_results)
