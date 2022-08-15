import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import MLP_MNIST.classe_rede_neural as nnc
import classe_rede_neural as nnc

import cv2 as cv2
import datetime as dt


L = 2
m = [2, 2, 2]
a = [1.7, 0.9]
b = [0.002, 1.]
num_classes = 2
#b = [2 / 3, 2 / 3, 2 / 3]
eta = [0.9, 0.8]
alpha = [0.0000,  0.]
#b = [2 / 3, 2 / 3, 2 / 3]



a1 = nnc.rede_neural(L, m, a, b)

data = {'y': [0, 1, 1, 0], 'x1': [-1, -1, 1, 1], 'x2': [-1, 1, -1, 1]}

dataset = pd.DataFrame(data=data)
test_dataset = dataset
print(dataset.head())

rnd_seed = np.random.seed(10)


n_epoch = 20
n_inst = len(dataset.index)
N = n_inst * n_epoch
step_plot = int(N / (n_epoch))

err_min = 0.5

a1, a1plt, Eav, n , acert = nnc.train_neural_network(a1, num_classes, rnd_seed, dataset, test_dataset, n_epoch, step_plot, eta, alpha,
                                              err_min)