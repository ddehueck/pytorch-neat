import torch
import gym
import numpy as np

from neat.phenotype.feed_forward import FeedForwardNet


class AcrobotBalanceConfig:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    NUM_INPUTS = 6
    NUM_OUTPUTS = 1
    USE_BIAS = True

    ACTIVATION = 'sigmoid'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = -1

    POPULATION_SIZE = 50
    NUMBER_OF_GENERATIONS = 100
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80

    TOP_HEIGHT = -np.inf

    #Allow episode lengths of > than 200
    gym.envs.register(
        id='Acrobot-v1',
        entry_point='gym.envs.classic_control:AcrobotEnv',
        max_episode_steps=200
    )

    #Fitness threshold increases as generations persist. Used for Acrobot
    def alt_fitness_fn(self, genome, generation):
        # OpenAI Gym
        env = gym.make('Acrobot-v1')
        done = False
        observation = env.reset()

        fitness = 0
        phenotype = FeedForwardNet(genome, self)
        counter = 0
        while not done:
            observation = np.array([observation])
            input = torch.Tensor(observation).to(self.DEVICE)
            pred = round(float(phenotype(input)))
            observation, reward, done, info = env.step(pred)
            #height = -np.cos(observation[0]) - np.cos(observation[1] + observation[0])
            #\cos(x+y)=\cos x\cos y\ +\sin x\sin y
            height = -observation[0] - (observation[0]*observation[2] - observation[1]*observation[3])
            fitness += height
        fitness = fitness/200
        print("fitness: ", fitness)
        env.close()

        return fitness
