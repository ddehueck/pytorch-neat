DEFAULT_KWARGS = { 

        'VERBOSE' : True,

        'NUM_INPUTS' : 784,
        'NUM_OUTPUTS' : 10,
        'USE_BIAS' : True, 

        'USE_CONV' : True,

        'GENERATIONAL_ENSEMBLE_SIZE' : 2,
        'CANDIDATE_LIMIT' : 1,

        'ACTIVATION' : 'sigmoid',
        'SCALE_ACTIVATION' : 4.9,

        'FITNESS_THRESHOLD' : float("inf"),

        'INITIAL_FITNESS_COEFFICIENT' : 0.1,
        'FINAL_FITNESS_COEFFICIENT' : 0.9,

        'POPULATION_SIZE' : 3,
        'NUMBER_OF_GENERATIONS' : 100,
        'SPECIATION_THRESHOLD' : 3.0,

        'CONNECTION_MUTATION_RATE' : 0.80,
        'CONNECTION_PERTURBATION_RATE' : 0.90,
        'ADD_NODE_MUTATION_RATE' : 0.03,
        'ADD_CONNECTION_MUTATION_RATE' : 0.5,

        'CROSSOVER_REENABLE_CONNECTION_GENE_RATE' : 0.25,

        'PERCENTAGE_TO_SAVE' : 0.80

}        
        
