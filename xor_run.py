import neat.neat as n
import neat.experiments.xor.config as c
from neat.visualize import draw_net
from tqdm import tqdm

num_of_solutions = 0

avg_num_hidden_nodes = 0
min_hidden_nodes = 100000
max_hidden_nodes = 0
found_minimal_solution = 0

avg_num_generations = 0
min_num_generations = 100000


for i in tqdm(range(1)):
    neat = n.Neat(c.XORConfig)
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

print('Total Number of Solutions: ', num_of_solutions)
print('Average Number of Hidden Nodes in a Solution', avg_num_hidden_nodes)
print('Solution found on average in:', avg_num_generations, 'generations')
print('Minimum number of hidden nodes:', min_hidden_nodes)
print('Maximum number of hidden nodes:', max_hidden_nodes)
print('Minimum number of generations:', min_num_generations)
print('Found minimal solution:', found_minimal_solution, 'times')
