import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import MLP_MNIST.classe_rede_neural as nnc
import classe_rede_neural as nnc

import cv2 as cv2
import datetime as dt
def plt_retas(rede,dataset, num_inst):
    # Realiza construção do gráfico 2D das entradas e as retas
    for n in range(0, num_inst):
        x1 = dataset.iloc[n, 1]
        x2 = dataset.iloc[n, 2]
        d = dataset.iloc[n, 0]
        plt.scatter(x1, x2, marker=f'${int(d)}$', s=200)
    x_space = np.linspace(-1, 1, 10)

    for j in range(0, rede.m[rede.L - 1]):
        b1 = rede.l[rede.L - 1].w[j][2]
        w1 = rede.l[rede.L - 1].w[j][0]
        w2 = rede.l[rede.L - 1].w[j][1]

        cy1 = (-b1 - w1 * x_space) / w2
        plt.plot(x_space, cy1)

    buttonPressed = False
    while not buttonPressed:
        buttonPressed = plt.waitforbuttonpress()

def plt_weights(a1, a1plt):
    wn0 = list()

    for l in range(0, a1.L):
        wn0.append(np.zeros((len(a1plt), a1.m[l] + 1)))
    for i in range(0, len(a1plt)):
        for l in range(0, a1.L):
            wn0[l][i] = a1plt[i].l[l].w[0]

    for l in range(0, a1.L):
        plt.figure(l)
        plt.plot(wn0[l])
        plt.title(f'Pesos do primeiro neurônio da camada {l}')
        plt.show()

def main():
    L = 2
    m = [2, 2, 2]
    a = [0.9, 0.9]
    b = [0.5, 0.5]
    #b = [2 / 3, 2 / 3, 2 / 3]
    eta = [0.9,0.8]
    alpha = [0.000,  0.]
    num_classes = 2


    a1 = nnc.rede_neural(L, m, a, b)

    data = {'y': [0, 1, 1, 0], 'x1': [-1, -1, 1, 1], 'x2': [-1, 1, -1, 1]}
    dataset = pd.DataFrame(data=data)
    test_dataset = dataset

    rnd_seed = np.random.seed(10)


    n_epoch = 2000
    n_inst = len(dataset.index)
    N = n_inst * n_epoch
    step_plot = int(N / (n_epoch))

    err_min = 0.05

    a1, a1plt, Eav, n , acert = nnc.train_neural_network(a1, num_classes, rnd_seed, dataset, test_dataset, n_epoch, step_plot, eta, alpha,
                                                  err_min)

    a1.save_neural_network('neural_network_XOR_BackProp.xlsx')
    plt_retas(a1,dataset,n_inst)

    plt.plot(acert)
    plt.title('Acertividade')

    buttonPressed = False
    while not buttonPressed:
        buttonPressed = plt.waitforbuttonpress()

    plt.plot(Eav)
    plt.title('Erro quadrático médio por época')

    buttonPressed = False
    while not buttonPressed:
        buttonPressed = plt.waitforbuttonpress()

    plt_weights(a1, a1plt)
    buttonPressed = False
    while not buttonPressed:
        buttonPressed = plt.waitforbuttonpress()

if __name__=='__main__':
  main()