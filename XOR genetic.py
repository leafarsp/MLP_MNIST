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
    num_inst = len(dataset.index)

    test_dataset = dataset


    num_individuos = 1000
    generations = 2000
    step_plot = 10
    err_min = 0.1
    target_fitness = 0.95
    mut_prob = 0.2
    mutation_multiplyer = 1.
    weight_limit = 10.
    #a1.initialize_weights_random()

    a1, best_fitness_plt, fitness_list, count_generations = nnc.train_genetic(
        a1, num_classes, rnd_seed, dataset,test_dataset, num_individuos, generations,
        step_plot, err_min, target_fitness, mut_prob, weight_limit, mutation_multiplyer)
    fitness_list.to_excel('fitness_list.xlsx')

    a1.save_neural_network("Xor_Genetic.xlsx")

    nnc.teste_neural_network(dataset,a1)

    plt.plot(best_fitness_plt[0:count_generations])
    buttonPressed = False
    while not buttonPressed:
        buttonPressed = plt.waitforbuttonpress()
    #plt_weights(a1,generations)
    plt_retas(a1, dataset, num_inst)



def plt_retas(rede,dataset, num_inst):
    # Realiza construção do gráfico 2D das entradas e as retas
    for n in range(0, num_inst):
        x1 = dataset.iloc[n, 1]
        x2 = dataset.iloc[n, 2]
        d = dataset.iloc[n, 0]
        plt.scatter(x1, x2, marker=f'${int(d)}$', s=200)
    x_space = np.linspace(-1, 1, 10)

    for j in range(0, rede.m[rede.L - 1]):
        b1 = rede.l[rede.L - 2].w[j][2]
        w1 = rede.l[rede.L - 2].w[j][0]
        w2 = rede.l[rede.L - 2].w[j][1]

        cy1 = (-b1 - w1 * x_space) / w2
        plt.plot(x_space, cy1)

    buttonPressed = False
    while not buttonPressed:
        buttonPressed = plt.waitforbuttonpress()

    # print(f'w[j]: {a1.l[1].w[j-1]}')

def plt_weights(rede,num_generations):
    wn0 = list()

    for l in range(0, rede.L):
        wn0.append(np.zeros((num_generations, rede.m[l] + 1)))
    for i in range(0, num_generations):
        for l in range(0, rede.L):
            wn0[l][i] = rede[i].l[l].w[0]

    for l in range(0, rede.L):
        plt.figure(l)
        plt.plot(wn0[l])
        plt.title(f'Pesos do primeiro neurônio da camada {l}')
        # plt.show()

    buttonPressed = False
    while not buttonPressed:
        buttonPressed = plt.waitforbuttonpress()

if __name__=='__main__':
  main()

