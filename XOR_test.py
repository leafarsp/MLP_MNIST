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
    eta = [0.9, 0.8]
    alpha = [0.0000,  0.]
    num_classes = 2

    a1 = nnc.rede_neural(L, m, a, b)

    data = {'y': [0, 1, 1, 0], 'x1': [-1, -1, 1, 1], 'x2': [-1, 1, -1, 1]}
    dataset = pd.DataFrame(data=data)
    test_dataset = dataset

    rnd_seed = np.random.seed(10)


    n_epoch = 1000
    n_inst = len(dataset.index)
    N = n_inst * n_epoch
    step_plot = int(N / (n_epoch))

    err_min = 0.05

    a1, a1plt, Eav, n , acert = nnc.train_neural_network(a1, num_classes, rnd_seed, dataset, test_dataset, n_epoch, step_plot, eta, alpha,
                                                  err_min)

    plt_results(a1, a1plt, Eav, dataset, n, acert)

    # err = nnc.calculate_err_epoch(dataset,a1,output_layer_activation)
    # print(f'erro:{err}')
    a1.save_neural_network('neural_network2.xlsx')


def plt_results(a1,a1plt,Eav, dataset, n, acert):
  # n_inst = len(dataset.index)
  # for l in range(0,a1.L):
  #   print(f'\n Layer {l}')
  #   print(a1.l[l].w)
  plt.figure(98)
  plt.plot(acert)
  plt.title('Acertividade')
  plt.show()

  #print(f'acertividade {teste_acertividade(dataset,a1)}%')

  plt.figure(99)
  plt.plot(Eav)
  plt.title('Erro quadrático médio por época')
  plt.show()



  wn0 = list()

  for l in range(0, a1.L):
    wn0.append(np.zeros((len(a1plt),a1.m[l]+1)))
  for i in range(0,len(a1plt)):
    for l in range(0, a1.L):
      wn0[l][i] = a1plt[i].l[l].w[0]



  for l in range(0, a1.L):
    plt.figure(l)
    plt.plot(wn0[l])
    plt.title(f'Pesos do primeiro neurônio da camada {l}')
    plt.show()



  # acertividade = teste_acertividade(dataset, a1)
  # plt.figure()
  # plt.title('Acertividade')
  # plt.plot(acertividade)

  plt.show()

  print(f'Épocas necessárias: {n}')





  buttonPressed = False
  while not buttonPressed:
    buttonPressed = plt.waitforbuttonpress()


if __name__=='__main__':
  main()