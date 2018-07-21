from v1.neat import Neat
from v1.experiments.xor import xor_fitness_fn
from v1.genotype.genome import Genome
from v1.visualize import draw_net

#g = Genome()
#g.add_connection_gene(0, 2)
#g.add_connection_gene(1, 2)

#draw_net(g, view=True, filename='./images/test-genomezzzz', show_disabled=True)


for i in range(100):
    print("ON ITERATION:::::", i)
    neat = Neat(150, 100, xor_fitness_fn)
    neat.run()
