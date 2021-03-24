import logging
import torch
import torch.nn as nn
import neat.activations as a
from torch import autograd


logger = logging.getLogger(__name__)


class FeedForwardNet(nn.Module):

    def __init__(self, genome, config):
        super(FeedForwardNet, self).__init__()
        self.genome = genome
        self.units = self.build_units()
        self.lin_modules = nn.ModuleList()
        self.config = config
        self.activation = a.Activations().get(config.ACTIVATION)

        for unit in self.units:
            self.lin_modules.append(unit.linear)

    def forward(self, x):
        outputs = dict()
        input_units = [u for u in self.units if u.ref_node.type == 'input']
        output_units = [u for u in self.units if u.ref_node.type == 'output']
        bias_units = [u for u in self.units if u.ref_node.type == 'bias']
        stacked_units = self.genome.order_units(self.units)

        # Set input values
        for u in input_units:
            outputs[u.ref_node.id] = x[0][u.ref_node.id]

        # Set bias value
        for u in bias_units:
            outputs[u.ref_node.id] = torch.ones((1, 1)).to(device)[0][0]

        # Compute through directed topology
        while len(stacked_units) > 0:
            current_unit = stacked_units.pop()

            if current_unit.ref_node.type != 'input' and current_unit.ref_node.type != 'bias':
                # Build input vector to current node
                inputs_ids = self.genome.get_inputs_ids(current_unit.ref_node.id)
                in_vec = autograd.Variable(torch.zeros((1, len(inputs_ids)), device=device, requires_grad=True))

                for i, input_id in enumerate(inputs_ids):
                    in_vec[0][i] = outputs[input_id]

                # Compute output of current node
                linear_module = self.lin_modules[self.units.index(current_unit)]
                if linear_module is not None:  # TODO: Can this be avoided?
                    scaled = self.config.SCALE_ACTIVATION * linear_module(in_vec)
                    out = self.activation(scaled)
                else:
                    out = torch.zeros((1, 1))

                # Add to outputs dictionary
                outputs[current_unit.ref_node.id] = out

        # Build output vector
        output = autograd.Variable(torch.zeros((1, len(output_units)), device=device, requires_grad=True))
        for i, u in enumerate(output_units):
            output[0][i] = outputs[u.ref_node.id]
        return output

    def build_units(self):
        units = []

        for n in self.genome.node_genes:
            in_genes = self.genome.get_connections_in(n.id)
            num_in = len(in_genes)
            weights = [g.weight for g in in_genes]

            new_unit = Unit(n, num_in)
            new_unit.set_weights(weights)

            units.append(new_unit)
        return units


class Unit:

    def __init__(self, ref_node, num_in_features):
        self.ref_node = ref_node
        self.linear = self.build_linear(num_in_features)

    def set_weights(self, weights):
        if self.ref_node.type != 'input' and self.ref_node.type != 'bias':
            weights = torch.cat(weights).unsqueeze(0)
            for p in self.linear.parameters():
                p.data = weights

    def build_linear(self, num_in_features):
        if self.ref_node.type == 'input' or self.ref_node.type == 'bias':
            return None
        return nn.Linear(num_in_features, 1, False)

    def __str__(self):
        return 'Reference Node: ' + str(self.ref_node) + '\n'


# TODO: Multiple GPU support get from config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
