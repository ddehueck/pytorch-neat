import torch
import gym
import numpy as np

from neat.phenotype.feed_forward import FeedForwardNet


class PoleBalanceConfig:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    NUM_INPUTS = 6
    NUM_OUTPUTS = 1
    USE_BIAS = True

    ACTIVATION = 'sigmoid'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 10000000.0

    POPULATION_SIZE = 150
    NUMBER_OF_GENERATIONS = 1000
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80

    # Allow episode lengths of > than 200
    # gym.envs.register(
    #     id='Acrobot-v1',
    #     entry_point='gym.envs.classic_control:AcrobotEnv',
    #     max_episode_steps=1000
    # )

    def fitness_fn(self, genome):
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
                # print(pred)
                observation, reward, done, info = env.step(pred)
                # print(reward)
                if counter % 100 == 0:
                    print(reward)
                reward = reward + 1
                counter += 1
                # print(reward)

                fitness += reward
                # print(fitness/)
            env.close()

            return fitness

