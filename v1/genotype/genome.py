import random
from v1.genotype.connection_gene import ConnectionGene
from v1.genotype.node_gene import NodeGene


class Genome:

    def __init__(self):
        self.connection_genes = []
        self.node_ids = set()
        self.innov_nums = set()
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

        if len(potential_outputs) is not 0 and len(potential_inputs) is not 0:
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
        new_node_id = len(self.node_ids)
        # Get a random existing connection
        existing_connection = self._get_rand_connection_gene()

        # Create two connections to replace existing connection with new node in the middle
        self.add_connection_gene(existing_connection.in_node_id, new_node_id, weight=1)
        self.add_connection_gene(new_node_id, existing_connection.out_node_id, weight=existing_connection.weight)

        # disable original connection
        existing_connection.is_enabled = False

    def build_nodes(self):
        nodes = []

        for n_id in self.node_ids:
            in_genes = self._get_connections_in(n_id)
            out_genes = self._get_connections_out(n_id)

            num_in = len(in_genes)
            num_out = len(out_genes)
            weights = [g.weight for g in in_genes]

            new_node = NodeGene(n_id, num_in, num_out, weights)
            nodes.append(new_node)
        return nodes

    def get_num_excess_genes(self, other):
        num_excess = 0

        max_innov_num = max(other.innov_nums)
        for c_gene in self.connection_genes:
            if c_gene.innov_num > max_innov_num:
                num_excess += 1

        return num_excess

    def get_num_disjoint_genes(self, other):
        num_disjoint = 0

        max_innov_num = max(other.innov_nums)
        for c_gene in self.connection_genes:
            if c_gene.innov_num <= max_innov_num:
                if other.get_connect_gene(c_gene.innov_num) is None:
                    num_disjoint += 1

        return num_disjoint

    def get_connect_gene(self, innov_num):
        for c_gene in self.connection_genes:
            if c_gene.innov_num == innov_num:
                return c_gene
        return None

    def get_avg_weight_difference(self, other):
        weight_difference = 0.0
        num_weights = 0.0

        for c_gene in self.connection_genes:
            matching_gene = other.get_connect_gene(c_gene.innov_num)
            if matching_gene is not None:
                weight_difference += float(c_gene.weight) - float(matching_gene.weight)
                num_weights += 1

        if num_weights == 0.0:
            num_weights = 1.0
        return weight_difference / num_weights

    def get_inputs_ids(self, node_id):
        """
        :param node_id: A node's id
        :return: An array of the ids of each node who's output is an input to the node_id param
        """
        node_input_ids = []
        for c_gene in self.connection_genes:
            if (c_gene.out_node_id == node_id) and c_gene.is_enabled:
                node_input_ids.append(c_gene.in_node_id)
        return node_input_ids

    def add_connection_gene(self, in_node_id, out_node_id, is_enabled=True, weight=None):
        new_c_gene = ConnectionGene(in_node_id, out_node_id, is_enabled)

        if weight is not None:
            new_c_gene.set_weight(float(weight))

        self.connection_genes.append(new_c_gene)
        # Maintain Genome attributes
        self.node_ids.add(in_node_id)
        self.node_ids.add(out_node_id)
        self.innov_nums.add(new_c_gene.innov_num)

    def add_connection_copy(self, copy):
        new_c_gene = ConnectionGene(copy.in_node_id, copy.out_node_id, copy.is_enabled)
        new_c_gene.set_weight(float(copy.weight))
        new_c_gene.set_innov_num(copy.innov_num)

        self.connection_genes.append(new_c_gene)
        # Maintain Genome attributes
        self.node_ids.add(copy.in_node_id)
        self.node_ids.add(copy.out_node_id)
        self.innov_nums.add(new_c_gene.innov_num)

    def _get_rand_node_id(self):
        return random.choice(list(self.node_ids))

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
        """
        Checks if the addition of a connection gene will create a cycle in the computation graph
        :param node_in_id: In node of the connection gene
        :param node_out_id: Out node of the connection gene
        :return: Boolean value
        """
        if node_in_id == node_out_id:
            return True

        visited = {node_out_id}
        while True:
            num_added = 0

            for c_gene in self.connection_genes:
                if c_gene.in_node_id in visited and c_gene.out_node_id not in visited:

                    if c_gene.out_node_id == node_in_id:
                        return True
                    else:
                        visited.add(c_gene.out_node_id)
                        num_added += 1

            if num_added == 0:
                return False

    def _is_valid_connection(self, node_in_id, node_out_id):
        does_creates_cycle = self.creates_cycle(node_in_id, node_out_id)
        does_connection_exist = self._does_connection_exist(node_in_id, node_out_id)

        return (not does_creates_cycle) and (not does_connection_exist)

    def _does_connection_exist(self, node_1_id, node_2_id):
        for c_gene in self.connection_genes:
            if (c_gene.in_node_id == node_1_id) and (c_gene.out_node_id == node_2_id):
                return True
            elif (c_gene.in_node_id == node_2_id) and (c_gene.out_node_id == node_1_id):
                return True
        return False

    def get_outputs(self, node, nodes):
        """
        Gets an unordered list of the node ids n_id outputs to
        :param node: The node who's output nodes are being retrieved
        :param nodes: List containing genome's node genes
        :return: List of node genes
        """
        out_ids = [c.out_node_id for c in self.connection_genes if (c.in_node_id == node.id) and c.is_enabled]
        return [n for n in nodes if n.id in out_ids]

    def order_nodes(self, nodes):
        """
        Implements a directed graph topological sort algorithm
        Requires an acyclic graph - see _is_valid_connection method
        :return: A sorted stack of NodeGene instances
        """

        visited = set()
        ordered = []

        for n in nodes:
            if n not in visited:
                self._order_nodes(n, nodes, ordered, visited)
        return ordered

    def _order_nodes(self, node, nodes, ordered, visited):
        visited.add(node)

        for out_node in self.get_outputs(node, nodes):
            if out_node not in visited:
                self._order_nodes(out_node, nodes, ordered, visited)

        ordered.append(node)

    def __str__(self):
        ret = 'Connections:\n\n'
        for connect_gene in self.connection_genes:
            ret += str(connect_gene) + '\n'
        return ret
