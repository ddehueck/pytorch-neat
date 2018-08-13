import torch.nn.functional as F


def sigmoid(x):
    return F.sigmoid(x)


def tanh(x):
    return F.tanh(x)


class Activations:

    def __init__(self):
        self.functions = dict(
            sigmoid=sigmoid,
            tanh=tanh
        )

    def get(self, func_name):
        return self.functions.get(func_name, None)
