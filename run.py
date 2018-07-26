import v1.neat as n
import v1.xor.config as c
from v1.visualize import draw_net
from tqdm import tqdm

num_of_solutions = 0
avg_num_hidden_nodes = 0
avg_num_generations = 0

for i in tqdm(range(100)):
    neat = n.Neat(c.XORConfig)
    solution, generation = neat.run()

    if solution is not None:
        avg_num_generations = ((avg_num_generations * num_of_solutions) + generation) / (num_of_solutions + 1)

        num_hidden_nodes = len([n for n in solution.build_nodes() if n.type == 'hidden'])
        avg_num_hidden_nodes = ((avg_num_hidden_nodes * num_of_solutions) + num_hidden_nodes) / (num_of_solutions + 1)

        num_of_solutions += 1
        draw_net(solution, view=True, filename='./images/solution-' + str(num_of_solutions), show_disabled=True)

print('Total Number of Solutions: ', num_of_solutions)
print('Average Number of Hidden Nodes in a Solution', avg_num_hidden_nodes)
print('Solution found on average in:', avg_num_generations, 'generations')
