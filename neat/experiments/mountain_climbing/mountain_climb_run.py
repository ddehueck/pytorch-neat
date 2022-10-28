import logging

import gym
import torch

import neat.population as pop
import neat.experiments.mountain_climbing.config as c
from neat.visualize import draw_net
from neat.phenotype.feed_forward import FeedForwardNet


logger = logging.getLogger(__name__)

logger.info(c.MountainClimbConfig.DEVICE)
neat = pop.Population(c.MountainClimbConfig)
solution, generation = neat.run()

if solution is not None:
    logger.info('Found a Solution')
    draw_net(solution, view=True, filename='./images/mountain-climb-solution', show_disabled=True)

    # OpenAI Gym
    env = gym.make('MountainCarContinuous-v0')
    done = False
    observation = env.reset()

    fitness = 0
    phenotype = FeedForwardNet(solution, c.MountainClimbConfig)

    while not done:
        env.render()
        input = torch.Tensor([observation]).to(c.MountainClimbConfig.DEVICE)

        pred = [round(float(phenotype(input)))]
        observation, reward, done, info = env.step(pred)

        fitness += reward
    env.close()
