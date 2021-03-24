import logging
import sys


logger = logging.getLogger(__name__)


class Species:

    def __init__(self, id, model_genome, generation):
        self.id = id
        self.model_genome = model_genome
        self.members = []
        self.fitness_history = []
        self.fitness = None
        self.adjusted_fitness = None
        self.last_improved = generation

    @staticmethod
    def species_distance(genome_1, genome_2):
        C1 = 1.0
        C2 = 1.0
        C3 = 0.5
        N = 1

        num_excess = genome_1.get_num_excess_genes(genome_2)
        num_disjoint = genome_1.get_num_disjoint_genes(genome_2)
        avg_weight_difference = genome_1.get_avg_weight_difference(genome_2)

        distance = (C1 * num_excess) / N
        distance += (C2 * num_disjoint) / N
        distance += C3 * avg_weight_difference

        return distance

    @staticmethod
    def stagnation(species, generation):
        """
        From https://github.com/CodeReclaimers/neat-python/neat/stagnation.py
        :param species: List of Species
        :param generation: generation number
        :return:
        """
        species_data = []
        for s in species:
            if len(s.fitness_history) > 0:
                prev_fitness = max(s.fitness_history)
            else:
                # super small number
                prev_fitness = -sys.float_info.max

            s.fitness = max([g.fitness for g in s.members])
            s.fitness_history.append(s.fitness)
            s.adjusted_fitness = None

            if prev_fitness is None or s.fitness >prev_fitness:
                s.last_improved = generation

            species_data.append(s)

        # sort in ascending fitness order
        species_data.sort(key=lambda g: g.fitness)

        result = []
        species_fitnesses = []
        num_non_stagnant = len(species_data)
        for i, s in enumerate(species_data):
            # Override stagnant state if marking this species as stagnant would
            # result in the total number of species dropping below the limit.
            # Because species are in ascending fitness order, less fit species
            # will be marked as stagnant first.
            stagnant_time = generation - s.last_improved
            is_stagnant = False
            if num_non_stagnant > 1:
                is_stagnant = stagnant_time >= 10

            if (len(species_data) - i) <= 1:
                is_stagnant = False

            if (len(species_data) - i) <= 1:
                is_stagnant = False

            if is_stagnant:
                num_non_stagnant -= 1

            result.append((s, is_stagnant))
            species_fitnesses.append(s.fitness)

        return result

