import torch
import numpy as np
import random
from v1.genotype.genome import Genome
from v1.species import Species
from copy import deepcopy


class Neat:
    __global_innovation_number = 0
    current_gen_innovation = []  # Can be reset after each generation according to paper

    def __init__(self, config):
        self.Config = config()
        self.population = self.set_initial_population()
        self.species = []

        for genome in self.population:
            self.speciate(genome, 0)

    def run(self):
        for generation in range(1, self.Config.NUMBER_OF_GENERATIONS):
            # Get Fitness of Every Genome
            for genome in self.population:
                genome.fitness = max(0, self.Config.fitness_fn(genome))

            best_genome = Neat.get_best_genome(self.population)

            # Reproduce
            all_fitnesses = []
            remaining_species = []

            for species, is_stagnant in Species.stagnation(self.species, generation):
                if is_stagnant:
                    self.species.remove(species)
                else:
                    all_fitnesses.extend(g.fitness for g in species.members)
                    remaining_species.append(species)

            min_fitness = min(all_fitnesses)
            max_fitness = max(all_fitnesses)

            fit_range = max(1.0, (max_fitness-min_fitness))
            for species in remaining_species:
                # Set adjusted fitness
                avg_species_fitness = np.mean([g.fitness for g in species.members])
                species.adjusted_fitness = (avg_species_fitness - min_fitness) / fit_range

            adj_fitnesses = [s.adjusted_fitness for s in remaining_species]
            adj_fitness_sum = sum(adj_fitnesses)

            # Get the number of offspring for each species
            new_population = []
            for species in remaining_species:
                if species.adjusted_fitness > 0:
                    size = max(2, int((species.adjusted_fitness/adj_fitness_sum) * self.Config.POPULATION_SIZE))
                else:
                    size = 2

                # sort current members in order of descending fitness
                cur_members = species.members
                cur_members.sort(key=lambda g: g.fitness, reverse=True)
                species.members = []  # reset

                # save top individual in species
                new_population.append(cur_members[0])
                size -= 1

                # Only allow top x% to reproduce
                purge_index = int(self.Config.PERCENTAGE_TO_SAVE * len(cur_members))
                purge_index = max(2, purge_index)
                cur_members = cur_members[:purge_index]

                for i in range(size):
                    parent_1 = random.choice(cur_members)
                    parent_2 = random.choice(cur_members)

                    child = self.crossover(parent_1, parent_2)
                    self.mutate(child)
                    new_population.append(child)

            # Set new population
            self.population = new_population
            Neat.current_gen_innovation = []

            # Speciate
            for genome in self.population:
                self.speciate(genome, generation)

            if best_genome.fitness >= self.Config.FITNESS_THRESHOLD:
                return best_genome, generation

            # Generation Stats
            if self.Config.VERBOSE:
                print('Finished Generation',  generation)
                print('Best Genome Fitness:', best_genome.fitness)
                print('Best Genome Length',   len(best_genome.connection_genes))
                print()

        return None, None

    def speciate(self, genome, generation):
        """
        Places Genome into proper species - index
        :param genome: Genome be speciated
        :param generation: Number of generation this speciation is occuring at
        :return: None
        """
        for species in self.species:
            if Species.species_distance(genome, species.model_genome) <= self.Config.SPECIATION_THRESHOLD:
                genome.species = species.id
                species.members.append(genome)
                return

        # Did not match any current species. Create a new one
        new_species = Species(len(self.species), genome, generation)
        genome.species = new_species.id
        new_species.members.append(genome)
        self.species.append(new_species)

    def assign_new_model_genomes(self, species):
        species_pop = self.get_genomes_in_species(species.id)
        species.model_genome = random.choice(species_pop)

    def get_genomes_in_species(self, species_id):
        return [g for g in self.population if g.species == species_id]

    def mutate(self, genome):
        """
        Applies connection and structural mutations at proper rate.
        Connection Mutations: Uniform Weight Perturbation or Replace Weight Value with Random Value
        Structural Mutations: Add Connection and Add Node
        :param genome: Genome to be mutated
        :return: None
        """

        if Neat.rand_uni_val() < self.Config.CONNECTION_MUTATION_RATE:
            for c_gene in genome.connection_genes:
                if Neat.rand_uni_val() < self.Config.CONNECTION_PERTURBATION_RATE:
                    perturb = Neat.rand_uni_val() * random.choice([1, -1])
                    c_gene.weight += perturb
                else:
                    c_gene.set_rand_weight()

        if Neat.rand_uni_val() < self.Config.ADD_NODE_MUTATION_RATE:
            genome.add_node_mutation()

        if Neat.rand_uni_val() < self.Config.ADD_CONNECTION_MUTATION_RATE:
            genome.add_connection_mutation()

    def set_initial_population(self): # TODO: Add to default config
        pop = []
        for i in range(self.Config.POPULATION_SIZE):
            new_genome = Genome()
            inputs = []
            outputs = []
            bias = None

            # Create nodes
            for j in range(self.Config.NUM_INPUTS):
                n = new_genome.add_node_gene('input')
                inputs.append(n)

            for j in range(self.Config.NUM_OUTPUTS):
                n = new_genome.add_node_gene('output')
                outputs.append(n)

            if self.Config.USE_BIAS:
                bias = new_genome.add_node_gene('bias')

            # Create connections
            for input in inputs:
                for output in outputs:
                    new_genome.add_connection_gene(input.id, output.id)

            if bias is not None:
                for output in outputs:
                    new_genome.add_connection_gene(bias.id, output.id)

            pop.append(new_genome)

        return pop

    def crossover(self, genome_1, genome_2):
        """
        Crossovers two Genome instances as described in the original NEAT implementation
        :param genome_1: First Genome Instance
        :param genome_2: Second Genome Instance
        :return: A child Genome Instance
        """

        child = Genome()
        best_parent, other_parent = Neat.order_parents(genome_1, genome_2)

        # Crossover connections
        # Randomly add matching genes from both parents
        for c_gene in best_parent.connection_genes:
            matching_gene = other_parent.get_connect_gene(c_gene.innov_num)

            if matching_gene is not None:
                # Randomly choose where to inherit gene from
                if Neat.rand_bool():
                    child_gene = deepcopy(c_gene)
                else:
                    child_gene = deepcopy(matching_gene)

            # No matching gene - is disjoint or excess
            # Inherit disjoint and excess genes from best parent
            else:
                child_gene = deepcopy(c_gene)

            # Apply rate of disabled gene being re-enabled
            if not child_gene.is_enabled:
                is_reenabeled = Neat.rand_uni_val() <= self.Config.CROSSOVER_REENABLE_CONNECTION_GENE_RATE
                enabled_in_best_parent = best_parent.get_connect_gene(child_gene.innov_num).is_enabled

                if is_reenabeled or enabled_in_best_parent:
                    child_gene.is_enabled = True

            child.add_connection_copy(child_gene)

        # Crossover Nodes
        # Randomly add matching genes from both parents
        for n_gene in best_parent.node_genes:
            matching_gene = other_parent.get_node_gene(n_gene.id)

            if matching_gene is not None:
                 # Randomly choose where to inherit gene from
                if Neat.rand_bool():
                    child_gene = deepcopy(n_gene)
                else:
                    child_gene = deepcopy(matching_gene)

            # No matching gene - is disjoint or excess
            # Inherit disjoint and excess genes from best parent
            else:
                child_gene = deepcopy(n_gene)

            child.add_node_copy(child_gene)

        return child

    @staticmethod
    def rand_uni_val():
        """
        Gets a random value from a uniform distribution on the interval [0, 1]
        :return: Float
        """
        return float(torch.rand(1))

    @staticmethod
    def rand_bool():
        return Neat.rand_uni_val() <= 0.5

    @staticmethod
    def get_best_genome(population):
        """
        Gets best genome out of a population
        :param population: List of Genome instances
        :return: Genome instance
        """
        population_copy = deepcopy(population)
        population_copy.sort(key=lambda g: g.fitness, reverse=True)

        return population_copy[0]

    @staticmethod
    def order_parents(parent_1, parent_2):
        """
        Orders parents with respect to fitness
        :param parent_1: First Parent Genome
        :param parent_2: Secont Parent Genome
        :return: Two Genome Instances
        """

        # genotype.cpp line 2043
        # Figure out which genotype is better
        # The worse genotype should not be allowed to add excess or disjoint genes
        # If they are the same, use the smaller one's disjoint and excess genes

        best_parent = parent_1
        other_parent = parent_2

        len_parent_1 = len(parent_1.connection_genes)
        len_parent_2 = len(parent_2.connection_genes)

        if parent_1.fitness == parent_2.fitness:
            if len_parent_1 == len_parent_2:
                # Fitness and Length equal - randomly choose best parent
                if Neat.rand_bool():
                    best_parent = parent_2
                    other_parent = parent_1
            # Choose minimal parent
            elif len_parent_2 < len_parent_1:
                best_parent = parent_2
                other_parent = parent_1

        elif parent_2.fitness > parent_1.fitness:
            best_parent = parent_2
            other_parent = parent_1

        return best_parent, other_parent

    @staticmethod
    def get_new_innovation_num():
        # Ensures that innovation numbers are being counted correctly
        # This should be the only way to get a new innovation numbers
        ret = Neat.__global_innovation_number
        Neat.__global_innovation_number += 1
        return ret
