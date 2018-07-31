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
        outputs = dict()
        input_nodes = [n for n in self.node_genes if n.type == 'input']
        output_nodes = [n for n in self.node_genes if n.type == 'output']
        stacked_nodes = self.genome.order_nodes(self.node_genes)

        # Set input values
        for n in input_nodes:
            if n.id == 2:
                # Is bias
                outputs[n.id] = torch.ones((1, 0)).to(device)[0]
            else:
                outputs[n.id] = x[0][n.id]

        # Compute through directed topology
        while len(stacked_nodes) > 0:
            current_node = stacked_nodes.pop()

            if current_node.type != 'input':
                # Build input vector to current node
                inputs_ids = self.genome.get_inputs_ids(current_node.id)
                in_vec = autograd.Variable(torch.zeros((1, len(inputs_ids)), device=device, requires_grad=True))

                for i, input_id in enumerate(inputs_ids):
                    in_vec[0][i] = outputs[input_id]

                # Compute output of current node
                linear_module = self.lin_modules[current_node.id]
                if linear_module is not None:  # TODO: Can this be avoided?
                    lin = max(torch.ones((1, 0))*-60.0, min(torch.ones((1, 0))*60.0, 5.0 * linear_module(in_vec)))
                    out = F.sigmoid(lin)
                    # Add to outputs dictionary
                else:
                    out = torch.zeros((1, 0))
                outputs[current_node.id] = out

        # Build output vector
        output = autograd.Variable(torch.zeros((1, len(output_nodes)), device=device, requires_grad=True))
        for i, n in enumerate(output_nodes):
            output[0][i] = outputs[n.id]
        return output


# TODO: Multiple GPU support
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
