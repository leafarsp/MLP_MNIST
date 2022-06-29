import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import classe_rede_neural as nnc

def main():
  L = 4
  m = [2, 512, 256, 128, 1]
  a = [0.9, 0.9, 0.9, 0.95]
  b = [1.,  1., 1., 1.]
  #b = [2 / 3, 2 / 3, 2 / 3]
  eta = [0.1,  0.2, 0.3, 0.8]
  alpha = [0., 0., 0., 0.]


  a1 = nnc.rede_neural(L, m, a, b)



  data = {'y': [0, 1, 1, 0], 'x1': [0, 0, 1, 1], 'x2': [0, 1, 0, 1]}
  dataset = pd.DataFrame(data=data)
  test_dataset = dataset

  rnd_seed = np.random.seed(10)

  n_epoch = 1000000
  n_inst = len(dataset.index)
  N = n_inst * n_epoch
  step_plot = int(N / (n_epoch/100))

  err_min = 1E-9
  a1, a1plt, Eav = nnc.train_neural_network(a1, rnd_seed, dataset, dataset, n_epoch, step_plot, eta, alpha, err_min)

  plt_results(a1, a1plt, Eav, dataset)



def plt_results(a1,a1plt,Eav, dataset):
  n_inst = len(dataset.index)
  for l in range(0,a1.L):
    print(f'\n Layer {l}')
    print(a1.l[l].w)

  print(f'acertividade {teste_rede(a1, dataset) * 100.}%')

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

  plt.figure()

  # Realiza construção do gráfico 2D das entradas e as retas
  for n in range(0, n_inst):
    x1 = dataset.iloc[n, 1]
    x2 = dataset.iloc[n, 2]
    d = dataset.iloc[n, 0]
    plt.scatter(x1, x2, marker=f'${int(d)}$', s=200)
  x_space = np.linspace(0, 1, 10)

  # percorre todos os neurônios da primeira camada oculta
  for j in range(0, a1.m[a1.L-1]):
    b1 = a1.l[a1.L-2].w[j][2]
    w1 = a1.l[a1.L-2].w[j][0]
    w2 = a1.l[a1.L-2].w[j][1]
    cy1 = (-b1 + w1 * x_space) / w2
    plt.plot(x_space, cy1)

  plt.show()

  print(f'Épocas necessárias: {n}')





  buttonPressed = False
  while not buttonPressed:
    buttonPressed = plt.waitforbuttonpress()


# Teste da rede, apresentando os dados de entrada e verificando a saída
def teste_rede(rede, dataset):
  count = 0

  n_inst = len(dataset.index)
  for i in range(0, n_inst):
    x = list(dataset.iloc[i, 1:])
    y_de = list(dataset.iloc[i])[0]

    y_rede = rede.forward_propagation(x=x)

    y_classificado = np.nan
    if (y_rede > 0.9):
      y_classificado = 1
    elif (y_rede < 0.1):
      y_classificado = 0
    print(f'{x} XOR -> {y_rede} valor considerado: y={y_classificado}')
    if y_classificado == y_de:
      count += 1
  return (count) / (n_inst)





if __name__=='__main__':
  main()