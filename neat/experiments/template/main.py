import json
import os
import sys

from neat.analysis import run_trial_analysis
from neat.experiments.template.config import TemplateConfig
from neat.population import Population

workdir = sys.argv[1]
params_file = os.path.join(workdir, "params.json")

with open(params_file, "r") as f:
    # Set up this trial
    params = json.load(f)
    config = TemplateConfig(**params)
    p = Population(config)

    #  Run the trial itself
    solution, generation = p.run()
    final_pop = p.population

    # Run analysis on the trial
    pred_map = config.create_activation_map(final_pop)
    ensemble_evaluator = config.ensemble_activations_evaluator
    df = run_trial_analysis(pred_map, ensemble_evaluator)
    df.to_csv('analysis_results.csv', encoding='utf-8')
