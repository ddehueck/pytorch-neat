import torch
import torch.nn as nn
from torch import autograd
from v1.temp.phenotype import FeedForwardNet
import v1.temp.visualize as viz


inputs = list(map(lambda s: autograd.Variable(torch.Tensor([s])), [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]))

targets = list(map(lambda s: autograd.Variable(torch.Tensor([s])), [
    [0],
    [1],
    [1],
    [0]
]))


def xor_fitness_fn(genome):
    fitness = 4.0  # Max fitness for XOR

    phenotype = FeedForwardNet(genome)
    phenotype.to(device)
    criterion = nn.MSELoss()

    for input, target in zip(inputs, targets):  # 4 training examples
        input, target = input.to(device), target.to(device)

        pred = phenotype(input)
        loss = criterion(pred, target)
        loss = float(loss)
        #loss = (float(pred) - float(target))**2

        fitness -= loss

    if fitness >= 3.9:
        print('----------------------------')
        print('SOOOLLLLLUUUUTTTTIIIIOOONNNN')
        print('----------------------------')
        print(genome)
        print('Fitness:', fitness)
        preds, labels = get_preds_and_labels(genome)
        print('Predictions:', str(preds))
        print('Labels:     ', str(labels))
        viz.draw_net(genome, view=True, show_disabled=True, filename='./images/solution')
        raise(Exception)
    return max(0, fitness)


def get_preds_and_labels(genome):
    phenotype = FeedForwardNet(genome)
    phenotype.to(device)

    predictions = []
    labels = []
    for input, target in zip(inputs, targets):  # 4 training examples
        input, target = input.to(device), target.to(device)

        predictions.append(float(phenotype(input)))
        labels.append(float(target))

    return predictions, labels


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
