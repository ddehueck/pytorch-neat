import torch
import random
from .visualize import draw_net
from .genotype.genome import Genome
from .species import Species
from copy import deepcopy
from .experiments.xor import get_preds_and_labels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Neat:

    def __init__(self, population_size, num_generations, fitness_fn):
        self.population_size = population_size
        self.population = self._set_population()
        self.species = []
        self.fitness_fn = fitness_fn
        self.num_generations = num_generations

    def run(self):
        for i in range(self.num_generations):
            new_population = []

            # Speciate
            for genome in self.population:
                self.speciate(genome)

            # Set fitnesses
            for genome in self.population:  # TODO: Run in parallel?
                genome.fitness = self.fitness_fn(genome)
                genome.adjusted_fitness = genome.fitness / len(self.get_genomes_in_species(genome.species))

            for species in self.species:
                species_pop = self.get_genomes_in_species(species.id)
                species.adjusted_fitness_sum = sum([g.adjusted_fitness for g in species_pop])

            # Reproduce
            all_adj_fitness_sum = sum([s.adjusted_fitness_sum for s in self.species])
            for species in self.species:
                new_species_size = int((species.adjusted_fitness_sum / all_adj_fitness_sum) * self.population_size)
                species_pop = self.get_genomes_in_species(species.id)

                if len(species_pop) >= 5:
                    # Maintain best genome
                    species_champ = Neat.get_best_genome(species_pop)
                    new_population.append(species_champ)
                    new_species_size -= 1

                    # Remove bottom 20% of population
                    Neat._sort_by_fitness(species_pop)
                    perc_20 = int(len(species_pop)*0.2)
                    species_pop = species_pop[:perc_20]

                for j in range(new_species_size):
                    parent_1 = random.choice(species_pop)
                    parent_2 = random.choice(species_pop)

                    child = self.crossover(parent_1, parent_2)
                    new_population.append(child)

            # Mutate children
            for genome in new_population:
                self.mutate(genome)

            # Save best network from this generation
            best_genome = Neat.get_best_genome(self.population)

            # Update for next generation
            self.population = new_population

            # Verbose
            print('------------------------')
            print("Completed generation:", i)
            print('------------------------')

            #draw_net(best_genome, view=True, show_disabled=True, filename='./images/best-genome-gen-' + str(i))
            print('Best Genome with a fitness of:', best_genome.fitness)
            print(best_genome)
            preds, labels = get_preds_and_labels(best_genome)
            print('Predictions:', str(preds))
            print('Labels:     ', str(labels))
            print('Num Species:', str(len(self.species)))

    @staticmethod
    def mutate(genome):
        # 80% chance of connections being mutated
        if Neat.weighted_rand_bool([0.8, 0.2]):
            # 90% chance of weights being uniformly perturbed 10% chance of being replaced with a rand value
            for connect_gene in genome.connection_genes:
                if Neat.weighted_rand_bool([0.9, 0.1]):
                    perturb = float(torch.rand((1, 0)))
                    if random.choice([True, False]):
                        perturb *= -1
                    connect_gene.weight += perturb
                else:
                    connect_gene.set_weight(float(torch.normal(torch.arange(0, 1))))

        # 3% chance of adding a new node
        if Neat.weighted_rand_bool([0.03, 0.97]):
            genome.add_node_mutation()

        # 5% chance of adding a connection
        if Neat.weighted_rand_bool([0.10, 0.90]):
            genome.add_connection_mutation()

    def speciate(self, genome):
        """
        Places Genome into proper species - index
        :param genome: Genome be speciated
        :return: None
        """
        THRESHOLD = 3.0  # TODO: Allow configuration
        for species in self.species:
            if Species.species_distance(genome, species.model_genome) <= THRESHOLD:
                genome.species = species.id
                return

        # Did not match any current species. Create a new one
        new_species = Species(len(self.species), genome)
        genome.species = new_species.id
        self.species.append(new_species)

        #TODO: Randomly assigned new generation individual as model species in self.species

    def get_genomes_in_species(self, species_id):
        genomes = []
        for individual in self.population:
            if individual.species == species_id:
                genomes.append(individual)
        return genomes

    @staticmethod
    def get_best_genome(population):
        best_genome = population[0]
        for genome in population:
            if genome.fitness > best_genome.fitness:
                best_genome = genome
        return best_genome

    @staticmethod
    def crossover(genome_1, genome_2):
        child = Genome()

        # genotype.cpp line 2043
        # Figure out which genotype is better
        # The worse genotype should not be allowed to add excess or disjoint genes
        # If they are the same, use the smaller one's disjoint and excess genes
        if genome_1.fitness == genome_2.fitness:
            if len(genome_1.connection_genes) == len(genome_2.connection_genes):
                # Fitness and Length equal - randomly choose best parent
                if random.choice([True, False]):
                    best_parent = genome_1
                    other_parent = genome_2
                else:
                    best_parent = genome_2
                    other_parent = genome_1
            elif len(genome_1.connection_genes) < len(genome_2.connection_genes):
                best_parent = genome_1
                other_parent = genome_2
            else:
                # Genome_2 less than genome_3
                best_parent = genome_2
                other_parent = genome_1
        elif genome_1.fitness > genome_2.fitness:
            best_parent = genome_1
            other_parent = genome_2
        else:
            best_parent = genome_2
            other_parent = genome_1

        # Randomly add matching genes from both parents
        for connect_gene in best_parent.connection_genes:
            if connect_gene.innov_num in other_parent.get_innov_nums():
                # Matching genes
                if random.choice([True, False]):
                    child_gene = deepcopy(connect_gene)
                    if not child_gene.is_enabled:
                        # 75% chance connection will be disabled if disabled
                        if Neat.weighted_rand_bool([0.25, 0.75]):
                            child_gene.is_enabled = True
                else:
                    child_gene = deepcopy(other_parent.get_connect_gene(connect_gene.innov_num))
                    if not child_gene.is_enabled:
                        # 75% chance connection will be disabled if disabled
                        if Neat.weighted_rand_bool([0.25, 0.75]):
                            child_gene.is_enabled = True

                child.connection_genes.append(child_gene)

        # Inherit disjoint and excess genes from best parent
        for connect_gene in best_parent.connection_genes:
            if connect_gene.innov_num not in other_parent.get_innov_nums():
                child_gene = deepcopy(connect_gene)
                if not child_gene.is_enabled:
                    # 75% chance connection will be disabled if disabled
                    if Neat.weighted_rand_bool([0.25, 0.75]):
                        child_gene.is_enabled = True

                child.connection_genes.append(child_gene)

        return child

    @staticmethod
    def weighted_rand_bool(weights):
        bool = random.choices(population=[True, False], weights=weights)[0]
        return bool

    def _set_population(self):
        pop = []
        for i in range(self.population_size):
            new_genome = Genome()
            new_genome.fitness = 4.0  # TODO: should be based on experiment
            new_genome.add_connection_gene(0, 3)
            new_genome.add_connection_gene(1, 3)
            new_genome.add_connection_gene(2, 3) # 2 is bias

            pop.append(new_genome)
        return pop

    @staticmethod
    def _sort_by_fitness(genome_list):
        """
        Returns highest to lowest from left to right within the list
        :param genome_list: List of genome objects
        :return: None - acts on param
        """
        genome_list.sort(key=lambda g: g.fitness, reverse=True)
