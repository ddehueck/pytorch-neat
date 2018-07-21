
class Species:

    def __init__(self, id, model_genome):
        self.id = id
        self.model_genome = model_genome
        self.adjusted_fitness_sum = None

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
