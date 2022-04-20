import logging

import neat.population as pop
import neat.experiments.chess.config as c
from neat.visualize import draw_net
from tqdm import tqdm

logger = logging.getLogger(__name__)

num_of_solutions = 0

avg_num_hidden_nodes = 0
min_hidden_nodes = 100000
max_hidden_nodes = 0
found_minimal_solution = 0

avg_num_generations = 0
min_num_generations = 100000


for i in tqdm(range(1)):
    neat = pop.Population(c.ChessConfig)
    solution, generation = neat.run()

    if solution is not None:
        avg_num_generations = ((avg_num_generations * num_of_solutions) + generation) / (num_of_solutions + 1)
        min_num_generations = min(generation, min_num_generations)

        num_hidden_nodes = len([n for n in solution.node_genes if n.type == 'hidden'])
        avg_num_hidden_nodes = ((avg_num_hidden_nodes * num_of_solutions) + num_hidden_nodes) / (num_of_solutions + 1)
        min_hidden_nodes = min(num_hidden_nodes, min_hidden_nodes)
        max_hidden_nodes = max(num_hidden_nodes, max_hidden_nodes)
        if num_hidden_nodes == 1:
            found_minimal_solution += 1

        num_of_solutions += 1
        draw_net(solution, view=True, filename='./images/solution-' + str(num_of_solutions), show_disabled=True)
        logger.info("Solution: {}".format(solution))
        logger.info("Generation: {}".format(generation))

logger.info('Total Number of Solutions: {}'.format(num_of_solutions))
logger.info('Average Number of Hidden Nodes in a Solution {}'.format(avg_num_hidden_nodes))
logger.info('Solution found on average in:{} generations'.format(avg_num_generations))
logger.info('Minimum number of hidden nodes: {}'.format(min_hidden_nodes))
logger.info('Maximum number of hidden nodes: {}'.format(max_hidden_nodes))
logger.info('Minimum number of generations: {}'.format(min_num_generations))
logger.info('Found minimal solution: {} times'.format(found_minimal_solution))
