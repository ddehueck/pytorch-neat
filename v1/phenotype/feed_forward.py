import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


class FeedForwardNet(nn.Module):

    def __init__(self, genome):
        super(FeedForwardNet, self).__init__()
        self.genome = genome
        self.connection_genes = genome.connection_genes
        self.node_genes = genome.build_nodes()
        self.lin_modules = nn.ModuleList()

        for gene in self.node_genes:
            self.lin_modules.append(gene.unit)

    def forward(self, x):
        # Need to follow proper connections here
        out_dict = dict()

        input_nodes = [node for node in self.node_genes if node.type == 'input']
        for node in input_nodes:
            # if not bias
            if node.id != 2:
                out_dict[node.id] = x[0][node.id]
            else:
                # give bias value
                out_dict[node.id] = torch.ones((1, 0)).to(device)[0]

        # Not every node will be able to be computed first time through
        non_input_nodes = [node for node in self.node_genes if node.type == 'hidden']
        while len(non_input_nodes) > 0:
            to_compute = []

            for node in non_input_nodes:
                input_nodes_ids = self.genome.get_nodes_input_nodes_ids(node.id)

                if self.unit_can_be_computed(input_nodes_ids, out_dict):
                    # Compute and set in out_dict
                    in_vec = autograd.Variable(torch.zeros((1, len(input_nodes_ids)), device=device, requires_grad=True))
                    for i, in_node_num in enumerate(input_nodes_ids):
                        in_vec[0][i] = out_dict[in_node_num]

                    out = F.sigmoid(self.lin_modules[node.id](in_vec))  # TODO: gene-unique activations
                    out_dict[node.id] = out
                else:
                    to_compute.append(node)

            non_input_nodes = to_compute

        output_nodes = [node for node in self.node_genes if node.type == 'output']
        output = autograd.Variable(torch.zeros((1, len(output_nodes)), device=device, requires_grad=True))
        for i, node in enumerate(output_nodes):
            input_nodes_ids = self.genome.get_nodes_input_nodes_ids(node.id)

            in_vec = autograd.Variable(torch.zeros((1, len(input_nodes_ids)), device=device, requires_grad=True))
            for j, in_node_num in enumerate(input_nodes_ids):
                in_vec[0][j] = out_dict[in_node_num]

            out = F.sigmoid(self.lin_modules[node.id](in_vec))  # TODO: gene-unique activations
            output[0][i] = out

        return output

    @staticmethod
    def unit_can_be_computed(input_node_ids, out_dict):
        return all(n_id in list(out_dict.keys()) for n_id in input_node_ids)


# TODO: Multiple GPU support
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
