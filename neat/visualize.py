import logging
import warnings

import graphviz


logger = logging.getLogger(__name__)


def draw_net(genome, view=False, filename=None, node_names=None, show_disabled=False, node_colors=None, fmt='png'):
    """ This is modified code originally from: https://github.com/CodeReclaimers/neat-python """
    """ Receives a genotype and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    for connect_gene in genome.connection_genes:
        if connect_gene.is_enabled or show_disabled:
            input = connect_gene.in_node_id
            output = connect_gene.out_node_id

            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))

            style = 'solid' if connect_gene.is_enabled else 'dotted'
            color = 'green' if float(connect_gene.weight) > 0 else 'red'
            width = str(0.1 + abs(float(connect_gene.weight / 5.0)))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot
