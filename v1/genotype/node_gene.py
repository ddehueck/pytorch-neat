import torch
import torch.nn as nn


class NodeGene:
    def __init__(self, node_id, num_in_connections, num_out_connections, weights):
        self.id = node_id
        self.type = self.get_type()
        self.unit = self._create_unit(num_in_connections, num_out_connections, weights)

    def _create_unit(self, num_in_features, num_out_features, weights):
        if self.type == 'input':
            return None
        elif self.type == 'output':
            num_out_features = 1

        unit = nn.Linear(num_in_features, num_out_features, False)
        weights = torch.cat(weights).unsqueeze(0)

        for p in unit.parameters():
            p.data = weights
        return unit

    def get_type(self):
        if self.id == 0 or self.id == 1 or self.id == 2:  # TODO: Make configurable
            node_type = 'input'
        elif self.id == 3:
            node_type = 'output'
        else:
            node_type = 'hidden'
        return node_type

    def __str__(self):
        return str(self.id) + '-' + self.type

