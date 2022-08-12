import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import classe_rede_neural as nnc

import cv2 as cv2
import datetime as dt

def main():
    L = 2
    m = [2, 2, 2]
    a = [0.9, 0.9]
    b = [0.5,0.5]
    #b = [2 / 3, 2 / 3, 2 / 3]

    num_classes = 2

    a1 = nnc.rede_neural(L, m, a, b)
    rnd_seed = 10

    a1.initialize_weights_random(rnd_seed)



    data = {'y': [0, 1, 1, 0], 'x1': [-1, -1, 1, 1], 'x2': [-1, 1, -1, 1]}
    dataset = pd.DataFrame(data=data)
    test_dataset = dataset


    num_individuos = 100
    generations = 500
    step_plot = 10
    err_min = 0.1
    target_fitness = 0.8
    mut_prob = 0.8
    a1.save_neural_network("Xor_Genetic.xlsx")
    #exit()
    a1 = nnc.train_genetic(a1, num_classes, rnd_seed, dataset, test_dataset, num_individuos, generations, step_plot, err_min, target_fitness, mut_prob)

    a1.save_neural_network("Xor_Genetic.xlsx")


if __name__=='__main__':
  main()

