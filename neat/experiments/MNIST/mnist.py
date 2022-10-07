import torch
import torch.nn as nn
from torch import autograd
from neat.phenotype.feed_forward import FeedForwardNet
# Import the MNIST dataset from torchvision
from torchvision import datasets, transforms

from matplotlib import pyplot as plt




class MNISTConfig:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    NUM_INPUTS = 2
    NUM_OUTPUTS = 1
    USE_BIAS = True

    ACTIVATION = 'sigmoid'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 3.9

    POPULATION_SIZE = 150
    NUMBER_OF_GENERATIONS = 150
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.30

    mnist_data = datasets.MNIST(root="./data", train=True, download=True)
    train = mnist_data.train_data
    # Convert to grayscale
    # train = transforms.Grayscale(num_output_channels=1)(train)
    train = train.view(train.size(0), -1).float()
    train = train / 255
    train_labels = mnist_data.train_labels
    test = mnist_data.test_data
    # test = transforms.Grayscale(num_output_channels=1)(test)
    test = test.view(test.size(0), -1).float()
    test = test / 255
    test_labels = mnist_data.test_labels
    
    # Convert to grayscale


    # Show the first image in the training set
    plt.imshow(train[0].view(28, 28))
    plt.show()

    # Print the shape of the train dataset
    print("Train shape:", train.shape)
    # Print the shape of the test dataset
    print("Test shape:", test.shape)
    

    # Split all of the examples into a python list
    inputs = torch.split(train, 1, dim=0)
    # print(len(inputs))
    # exit()
    targets = torch.split(train_labels, 1, dim=0)
    # print(len(targets))
    # exit()

    # targets = list(map(lambda s: torch.Tensor([s]), [[0],[1],[1],[0]]))
    # Print the targets
    print(targets)

    def fitness_fn(self, genome):
        fitness = 4.0  # Max fitness for XOR

        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)
        criterion = nn.MSELoss()

        for input, target in zip(self.inputs, self.targets):  # 4 training examples
            input, target = input.to(self.DEVICE), target.to(self.DEVICE)

            pred = phenotype(input)
            loss = (float(pred) - float(target)) ** 2
            loss = float(loss)

            fitness -= loss
            # loss = criterion(pred, target)

        return fitness

    def get_preds_and_labels(self, genome):
        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)

        predictions = []
        labels = []
        for input, target in zip(self.inputs, self.targets):  # 4 training examples
            input, target = input.to(self.DEVICE), target.to(self.DEVICE)

            predictions.append(float(phenotype(input)))
            labels.append(float(target))

        return predictions, labels
