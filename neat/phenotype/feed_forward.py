import torch
import torch.nn as nn

import neat.activations as a


dtype = torch.float64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FeedForwardNet(nn.Module):

    def __init__(self, genome, config):
        super(FeedForwardNet, self).__init__()
        self.genome = genome
        self.config = config
        self.units = self.build_units()
        self.lin_modules = nn.ModuleList()
        self.activation = a.Activations().get(config.ACTIVATION)
        self.bias_indices = [u.ref_node.id for u in self.units if u.is_bias()]
        self.input_indices = torch.tensor([u.ref_node.id for u in self.units if u.is_input()], device=device)
        self.output_indices = torch.tensor([u.ref_node.id for u in self.units if u.is_output()], device=device)
        self.stacked_units = self.genome.order_units(self.units)
        self.units_indexes = {u.ref_node.id: k for k, u in enumerate(self.units)}
        self.units_input_indices = [
            torch.tensor(self.genome.get_inputs_ids(u.ref_node.id), dtype=torch.long, device=device) for u in self.units
        ]
        self.extras_len = len(self.units) - len(self.input_indices)
        self.build_modules()

    def forward(self, x):
        inputs = torch.index_select(x, 1, self.input_indices)
        extras = torch.ones((1, self.extras_len), dtype=dtype, device=device)
        output_tensor = torch.cat((inputs, extras), 1)
        # assert output_tensor.size(1) == len(self.units)

        # Compute through directed topology
        for current_unit in reversed(self.stacked_units):
            if not current_unit.is_input() and not current_unit.is_bias():
                # Get unit output index.
                unit_index = self.units_indexes[current_unit.ref_node.id]
                # Build input vector to current node
                inputs_ids = self.units_input_indices[unit_index]
                in_vec = torch.index_select(output_tensor, 1, inputs_ids)
                # Compute output of current node
                scaled = self.config.SCALE_ACTIVATION * current_unit.linear(in_vec)
                # Set output after activation
                output_tensor[0][unit_index] = self.activation(scaled)
        return torch.index_select(output_tensor, 1, self.output_indices)

    def build_modules(self):
        for unit in self.units:
            self.lin_modules.append(unit.linear)

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

    def is_input(self):
        return self.ref_node.type == 'input'

    def is_output(self):
        return self.ref_node.type == 'output'

    def is_bias(self):
        return self.ref_node.type == 'bias'

    def is_hidden(self):
        return self.ref_node.type == 'hidden'

    def set_weights(self, weights):
        if not self.is_input() and not self.is_bias():
            weights = torch.cat(weights).unsqueeze(0)
            for p in self.linear.parameters():
                p.data = weights

    def build_linear(self, num_in_features):
        if self.is_input() or self.is_bias():
            return None
        return nn.Linear(num_in_features, 1, False)

    def __str__(self):
        return 'Reference Node: {}\n'.format(self.ref_node)
