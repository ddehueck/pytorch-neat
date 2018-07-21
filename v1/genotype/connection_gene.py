import torch
import v1.genotype.genome

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConnectionGene:

    def __init__(self, in_node_id, out_node_id, is_enabled):
        self.in_node_id = in_node_id
        self.out_node_id = out_node_id
        self.is_enabled = is_enabled
        self.weight = torch.Tensor(torch.normal(torch.arange(0, 1))).to(device)
        self.innov_num = self._get_correct_innovation_num()

    def set_weight(self, new_weight):
        """
        Sets new weight
        :param new_weight: type float - not tensor
        :return: None
        """
        self.weight = torch.Tensor([new_weight]).to(device)

    def _get_correct_innovation_num(self):
        # This method keeps track of a generation's innovations
        for connect_gene in v1.genotype.genome.Genome.current_gen_innovation:
            if (connect_gene.in_node_id == self.in_node_id) and (connect_gene.out_node_id == self.out_node_id):
                return connect_gene.innov_num  # TODO: Look into def __eq__(self, other): and others
        # New innovation
        v1.genotype.genome.Genome.current_gen_innovation.append(self)
        return v1.genotype.genome.Genome.get_new_innovation_num()

    def __str__(self):
        return 'In: ' + str(self.in_node_id) + '\nOut: ' + str(self.out_node_id) + '\nIs Enabled: ' + str(self.is_enabled) + '\nInnovation #: ' + str(self.innov_num) + '\n'
