import gym
import torch
import v1.neat as n
import v1.experiments.pole_balancing.config as c
from v1.visualize import draw_net
from v1.phenotype.feed_forward import FeedForwardNet

neat = n.Neat(c.PoleBalanceConfig)
solution, generation = neat.run()

if solution is not None:
    print('Found a Solution')
    draw_net(solution, view=True, filename='./images/pole-balancing-solution', show_disabled=True)

    # OpenAI Gym
    env = gym.make('LongCartPole-v0')
    done = False
    observation = env.reset()

    fitness = 0
    phenotype = FeedForwardNet(solution, c.PoleBalanceConfig)

    while not done:
        env.render()
        input = torch.Tensor([observation]).to(c.PoleBalanceConfig.DEVICE)

        pred = round(float(phenotype(input)))
        observation, reward, done, info = env.step(pred)

        fitness += reward
    env.close()
