import logging

import neat.population as pop
import neat.experiments.UCI.config as c
from neat.experiments.UCI.kwargs import KWARGS

from neat.visualize import draw_net
from tqdm import tqdm

import uci_dataset as uci
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.nn.functional import one_hot

logger = logging.getLogger(__name__)



df = uci.load_heart_disease().dropna()

features = df.iloc[:,:-1]
target = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = torch.tensor(scaler.transform(X_train))
X_test = torch.tensor(scaler.transform(X_test))

y_train = torch.squeeze(one_hot(torch.tensor(y_train.to_numpy().reshape(-1,1))))  # type: ignore
y_test = torch.squeeze(one_hot(torch.tensor(y_test.to_numpy().reshape(-1,1)))) # type: ignore

kwargs = KWARGS
kwargs['DATA'] = X_train
kwargs['TARGET'] = y_train
kwargs['NUM_INPUTS'] = kwargs['DATA'].shape[1]
kwargs['NUM_OUTPUTS'] = kwargs['TARGET'].shape[1]



neat = pop.Population(c.UCIConfig(**kwargs))
solution, generation = neat.run()

