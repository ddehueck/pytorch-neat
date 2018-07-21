import torch
import random
import v1.genotype.connection_gene as cg
from .node_gene import NodeGene
import v1.visualize as viz


class Genome:
    __global_innovation_number = 0
    current_gen_innovation = []  # Can be reset after each generation according to paper

    def __init__(self):
        self.connection_genes = []
        self.fitness = None
        self.adjusted_fitness = None
        self.species = None

    def add_connection_mutation(self):
        """
        In the add connection mutation, a single new connection gene is added
        connecting two previously unconnected nodes.
        """
        # TODO: better algorithm?

        all_nodes = self.build_nodes()
        potential_inputs = [n.id for n in all_nodes if n.type != 'output']
        potential_outputs = [n.id for n in all_nodes if n.type != 'input']

        node_in_id = random.choice(potential_inputs)
        node_out_id = random.choice(potential_outputs)

        if self._is_valid_connection(node_in_id, node_out_id):
            # Add connection
            self.add_connection_gene(node_in_id, node_out_id)

    def add_node_mutation(self):
        """
        This method implicitly adds a node by modifying connection genes.
         In the add node mutation an existing connection is split and the new node placed where the old
         connection used to be. The old connection is disabled and two new connections are added to the genotype.
         The new connection leading into the new node receives a weight of 1, and the new connection leading
         out receives the same weight as the old connection.
        """
        # Get new_node id
        new_node_id = len(self._get_node_ids())
        # Get a random existing connection
        existing_connection = self._get_rand_connection_gene()

        # Create two connections to replace existing connection with new node in the middle
        self.add_connection_gene(existing_connection.in_node_id, new_node_id, weight=1)
        self.add_connection_gene(new_node_id, existing_connection.out_node_id, weight=existing_connection.weight)

        # disable original connection
        #existing_connection.is_enabled = False
        # Remove it from genome
        self.connection_genes.remove(existing_connection)

    def build_nodes(self):
        node_ids = self._get_node_ids()
        nodes = []

        for n_id in node_ids:
            in_genes = self._get_connections_in(n_id)
            out_genes = self._get_connections_out(n_id)

            num_in = len(in_genes)
            num_out = len(out_genes)
            weights = [g.weight for g in in_genes]

            new_node = NodeGene(n_id, num_in, num_out, weights)
            nodes.append(new_node)
        return nodes

    def get_num_excess_genes(self, compare_to_genome):
        num_excess = 0

        max_innov_num = max(compare_to_genome.get_innov_nums())
        for connect_gene in self.connection_genes:
            if connect_gene.innov_num > max_innov_num:
                num_excess += 1

        return num_excess

    def get_num_disjoint_genes(self, compare_to_genome):
        num_disjoint = 0

        max_innov_num = max(compare_to_genome.get_innov_nums())
        for connect_gene in self.connection_genes:
            if connect_gene.innov_num <= max_innov_num:
                if compare_to_genome.get_connect_gene(connect_gene.innov_num) is None:
                    num_disjoint += 1

        return num_disjoint

    def get_innov_nums(self):
        return [gene.innov_num for gene in self.connection_genes]

    def get_connect_gene(self, innov_num):
        for gene in self.connection_genes:
            if gene.innov_num == innov_num:
                return gene
        return None

    def get_avg_weight_difference(self, compare_to_genome):
        weight_difference = 0.0
        num_weights = 0.0

        for connect_gene in self.connection_genes:
            matching_gene = compare_to_genome.get_connect_gene(connect_gene.innov_num)
            if matching_gene is not None:
                weight_difference += float(connect_gene.weight) - float(matching_gene.weight)
                num_weights += 1

        if num_weights == 0.0:
            num_weights = 1.0
        return weight_difference / num_weights

    def get_nodes_input_nodes_ids(self, node_id):
        """
        :param node_id: A node's id
        :return: An array of the ids of each node who's output is an input to the node_id param
        """
        node_input_ids = []
        for gene in self.connection_genes:
            if (gene.out_node_id == node_id) and gene.is_enabled:
                node_input_ids.append(gene.in_node_id)
        return node_input_ids

    def add_connection_gene(self, in_node_id, out_node_id, is_enabled=True, weight=torch.normal(torch.arange(0, 1))):
        new_connection_gene = cg.ConnectionGene(in_node_id, out_node_id, is_enabled)
        new_connection_gene.set_weight(float(weight))
        self.connection_genes.append(new_connection_gene)

    def _get_node_ids(self):
        # TODO: Make this a maintained class variable instead of being computed each time?
        """
        :return: A set of a node ids referenced in self.connection_nodes
        """
        node_ids = set()
        for gene in self.connection_genes:
            node_ids.add(gene.in_node_id)
            node_ids.add(gene.out_node_id)
        return node_ids

    def _get_rand_node_id(self):
        node_ids = list(self._get_node_ids())
        return random.choice(node_ids)

    def _get_rand_connection_gene(self):
        return random.choice(self.connection_genes)

    def _get_connections_in(self, node_id):
        """
        :return: the connection genes in to the node identified by the :param: node_id
        """
        genes = []
        for gene in self.connection_genes:
            if (gene.out_node_id == node_id) and gene.is_enabled:
                genes.append(gene)
        return genes

    def _get_connections_out(self, node_id):
        """
        :return: the connection genes out of the node identified by the :param: node_id
        """
        genes = []
        for gene in self.connection_genes:
            if (gene.in_node_id == node_id) and gene.is_enabled:
                genes.append(gene)
        return genes

    def creates_cycle(self, node_in_id, node_out_id):
        if node_in_id == node_out_id:
            return True

        visited = {node_out_id}
        while True:
            num_added = 0

            for gene in self.connection_genes:
                if gene.in_node_id in visited and gene.out_node_id not in visited:

                    if gene.out_node_id == node_in_id:
                        return True
                    else:
                        visited.add(gene.out_node_id)
                        num_added += 1

            if num_added == 0:
                return False

    def _is_valid_connection(self, node_in_id, node_out_id):
        creates_cycle = self.creates_cycle(node_in_id, node_out_id)
        return (not creates_cycle) and (not self._does_connection_exist(node_in_id, node_out_id))

    def _does_connection_exist(self, node_1_id, node_2_id):
        for connect_gene in self.connection_genes:
            if (connect_gene.in_node_id == node_1_id) and (connect_gene.out_node_id == node_2_id):
                return True
            elif (connect_gene.in_node_id == node_2_id) and (connect_gene.out_node_id == node_1_id):
                return True
        return False

    @staticmethod
    def get_new_innovation_num():
        # Ensures that innovation numbers are being counted correctly
        # This should be the only way to get a new innovation numbers
        ret = Genome.__global_innovation_number
        Genome.__global_innovation_number += 1
        return ret

    def __str__(self):
        ret = 'Connections:\n\n'
        for connect_gene in self.connection_genes:
            ret += str(connect_gene) + '\n'
        return ret