
class NodeGene:
    def __init__(self, node_id, node_type):
        self.id = node_id
        self.type = node_type
        self.unit = None

    def __str__(self):
        return str(self.id) + '-' + self.type

