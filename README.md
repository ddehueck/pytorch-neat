# PyTorch-NEAT
A PyTorch implementation of the NEAT (NeuroEvolution of Augmenting Topologies) method which was originally created by Kenneth O. Stanley as a principled approach to evolving neural networks. [Read the paper here](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf).

### Experiments
PyTorch-NEAT currently contains three built-in experiments: XOR, Single-Pole Balancing, and Car Mountain Climbing.

##### XOR Experiment
Run with the command: ```python xor_run.py```
Will run up-to 150 generations with an initial population of 150 genomes. When/If a solution is found the solution network will be displayed along with statistics about the trial. Feel free to run for more than one trial - just increase the range in the outer for loop in the xor_run.py file.

##### Single Pole Balancing
Run with the command: ```python pole_run.py```
Will run up-to 150 generations with an initial population of 150 genomes. Runs in the OpenAI gym enviornment. When/If a solution is found the solution network will be displayed along with a rendered evalution in the OpenAI gym.

##### Car Mountain Climbing Experiment
Run with the command: ```python mountain_climb_run.py```
Will run up-to 150 generations with an initial population of 150 genomes. Runs in the OpenAI gym enviornment. When/If a solution is found the solution network will be displayed along with a rendered evalution in the OpenAI gym.
### An Experiment's Configuration File
Each experiment requries a configuration file. The XOR experiment config file is broken down here:

Import necessary items.
```python
import torch
import torch.nn as nn
from torch import autograd
from v1.phenotype.feed_forward import FeedForwardNet
```

A config file consists of a Python class with certain requirnments (detailed in comments below).
```python
class XORConfig:
    # Where to evaluate tensors
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Boolean - print generation stats throughout trial
    VERBOSE = False

    # Number of inputs/outputs each genome should contain
    NUM_INPUTS = 2
    NUM_OUTPUTS = 1
    # Boolean - use a bias node in each genome
    USE_BIAS = True
    
    # String - which activation function each node will use
    # Note: currently only sigmoid and tanh are available - see v1/activations.py for functions
    ACTIVATION = 'sigmoid'
    # Float - what value to scale the activation function's input by
    # This default value is taken directly from the paper
    SCALE_ACTIVATION = 4.9
    
    # Float - a solution is defined as having a fitness >= this fitness threshold
    FITNESS_THRESHOLD = 3.9

    # Integer - size of population
    POPULATION_SIZE = 150
    # Integer - max number of generations to be run for
    NUMBER_OF_GENERATIONS = 150
    # Float - an organism is said to be in a species if the genome distance to the model genome of a species is <= this speciation threshold
    SPECIATION_THRESHOLD = 3.0

    # Float between 0.0 and 1.0 - rate at which a connection gene will be mutated
    CONNECTION_MUTATION_RATE = 0.80
    # Float between 0.0 and 1.0 - rate at which a connections weight is perturbed (if connection is to be mutated) 
    CONNECTION_PERTURBATION_RATE = 0.90
    # Float between 0.0 and 1.0 - rate at which a node will randomly be added to a genome
    ADD_NODE_MUTATION_RATE = 0.03
    # Float between 0.0 and 1.0 - rate at which a connection will randomly be added to a genome
    ADD_CONNECTION_MUTATION_RATE = 0.5
    
    # Float between 0.0 and 1.0 - rate at which a connection, if disabled, will be re-enabled
    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Float between 0.0 and 1.0 - Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.30
    
    # XOR's input and output values
    # Note: it is not always necessary to explicity include these values. Depends on the fitness evaluation.
    # See an OpenAI gym experiment config file for a different fitness evaluation example.
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
```

It is **required** for an experiment's configuration class to contain a ```fitness_fn()``` method. It takes just one argument - a genome.

```python
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

        return fitness
```
Feel free to add additional methods for experiment-specific uses.
```python
    def get_preds_and_labels(self, genome):
        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)

        predictions = []
        labels = []
        for input, target in zip(self.inputs, self.targets):  # 4 training examples
            input, target = input.to(self.DEVICE), target.to(self.DEVICE)

            predictions.append(float(phenotype(input)))
            labels.append(float(target))
```
### Contributors
* [Devin de Hueck](https://ddehueck.github.io/)
* [Justin Chen](https://github.com/ch3njust1n)
* [Lucia Vilallonga](https://github.com/ghostpress)

## License: MIT

Copyright (c) 2018
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
