import torch
import neat.population
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConnectionGene:

    def __init__(self, in_node_id, out_node_id, is_enabled):
        self.in_node_id = in_node_id
        self.out_node_id = out_node_id
        self.is_enabled = is_enabled
        self.innov_num = self._get_correct_innovation_num()
        self.weight = None

        self.set_rand_weight()

    def set_weight(self, new_weight):
        """
        Sets new weight
        :param new_weight: type float
        :return: None
        """
        self.weight = torch.Tensor([new_weight]).to(device)

    def set_rand_weight(self):
        """
        Weight is set to a random value
        :return: None - modifies object
        """
        self.weight = torch.Tensor(torch.normal(torch.arange(0, 1).float())).to(device)

    def set_innov_num(self, num):
        """
        Only use when copying a gene to avoid speciation issues
        :return: None - modifies object
        """
        self.innov_num = num

    def _get_correct_innovation_num(self):
        # This method keeps track of a generation's innovations
        for connect_gene in neat.population.Population.current_gen_innovation:
            if self == connect_gene:
                return connect_gene.innov_num
        # Is new innovation
        neat.population.Population.current_gen_innovation.append(self)
        return neat.population.Population.get_new_innovation_num()

    def __eq__(self, other):
        return (self.in_node_id == other.in_node_id) and (self.out_node_id == other.out_node_id)

    def __str__(self):
        return 'In: ' + str(self.in_node_id) + '\nOut: ' + str(self.out_node_id) + '\nIs Enabled: ' + str(self.is_enabled) + '\nInnovation #: ' + str(self.innov_num) + '\nWeight: ' + str(float(self.weight)) + '\n'
