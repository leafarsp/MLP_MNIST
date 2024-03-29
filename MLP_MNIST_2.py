import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import classe_rede_neural as nnc
import cv2 as cv2



def main():
    L = 2
    m = [(28 * 28), 15, 10]
    # a = [1.7, 0.9]
    # b = [0.002, 1.]
    a = [1., 1.]
    b = [1., 1.]

    #b = [2 / 3, 2 / 3, 2 / 3]
    eta = [0.1, 0.1]
    learning_rate_end = eta[0]
    alpha = [0.05,  0.05]


    #
    # a1 = nnc.rede_neural(L, m, a, b)

    # a1 = nnc.load_neural_network('MNIST_genetic\\MNIST_genetic_0000.xlsx')


    a1 = nnc.load_neural_network('MNISTS_BackProp.xlsx')

    # Base de dados de treinamento
    # Se for utilizar o Jupyter notebook, utilizar a linha abaixo
    # dataset = pd.read_csv('mnist_test.csv')
    dataset = pd.read_csv('mnist_train_small.csv')


    #Filtrando apenas o número 1
    # dataset = dataset.loc[dataset['7'] == 1]
    #dataset = dataset[dataset['6'].isin([1,2,3,4,5,6,7,8,9,0])]

    dataset = dataset.iloc[0:]

    dataset.iloc[:, 1:] = dataset.iloc[:, 1:] / 255
    dataset.iloc[:, 1:] = dataset.iloc[:, 1:] * 2. - 1.


    test_dataset = pd.read_csv('mnist_test.csv')
    test_dataset = test_dataset.iloc[0:]
    test_dataset.iloc[:, 1:] = test_dataset.iloc[:, 1:] / 255
    test_dataset.iloc[:, 1:] = test_dataset.iloc[:, 1:] * 2. - 1.

    dataset.head()

    rnd_seed = 10

    n_epoch = 100
    n_inst = len(dataset.index)
    N = n_inst * n_epoch
    step_plot = int(N / (n_epoch * 1))

    err_min = 0.5
    # a1, a1plt, Eav, n = nnc.train_neural_network(a1, rnd_seed, dataset, dataset, n_epoch, step_plot, eta, alpha, err_min)
    #
    # a1.save_neural_network()
    #a1 = nnc.load_neural_network('neural_network2.xlsx')
    # print(f'\na2.l[l].w=\n{a2.l[1].w}')
    #

    a1, a1plt, Eav, n , acert = nnc.train_neural_network(
      rede=a1,
      num_classes=10,
      rnd_seed=rnd_seed,
      dataset=dataset,
      test_dataset=test_dataset,
      n_epoch=n_epoch,
      step_plot=step_plot,
      learning_rate=eta,
      momentum=alpha,
      err_min=err_min,
      weight_limit=1.,
      learning_rate_end=learning_rate_end)



    a1.save_neural_network('MNISTS_BackProp.xlsx')
    # print(f'\na1.l[l].w=\n{a1.l[1].w}')

    plt_results(a1, a1plt, Eav, dataset, n, acert)
    a1.flag_test_acertividade = False
    err = nnc.calculate_err_epoch(dataset,a1,output_layer_activation)
    print(f'erro:{err}')

    acertividade = nnc.teste_acertividade(dataset, 10, a1, True)
    print(f'Acertividade: {a1.acertividade:.3f}%')

def output_layer_activation(output_value):
  d = np.ones(10) * -1
  #num = dataset_shufle.iloc[ni, 0]
  d[output_value] = 1
  return d

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
  # print(f'0 XOR 0 = {a1.forward_propagation(x=[0, 0])}')
  # print(f'0 XOR 1 = {a1.forward_propagation(x=[0, 1])}')
  # print(f'1 XOR 0 = {a1.forward_propagation(x=[1, 0])}')
  # print(f'1 XOR 1 = {a1.forward_propagation(x=[1, 1])}')

def digit_recog(rede, image_array=None, training_instance=0, dataset=None):
    # progagação do sinal forward


    if type(image_array) == type(None):
      image_array = list(dataset.iloc[training_instance, 1:(rede.m[0] + 1)])
    # Acrescenta 1 que é relativo ao bias

    y_out = rede.forward_propagation(image_array)

    num_out = np.nan
    cont_neuronio_ativo = 0
    for j in range(0, rede.m[rede.L]):
      if (y_out[j] > (1 * 0.8)):
        num_out = j
        cont_neuronio_ativo += 1
      if (cont_neuronio_ativo > 1):
        num_out = np.nan
        break

    return num_out, y_out


def display_number(dataset, dataset_position, save=False):
  im_data = np.zeros((28, 28, 1))
  # im_data = dataset.iloc[0,1:65]
  for i in range(0, 28):
    im_data[i, :, 0] = dataset.iloc[dataset_position, 28 * (i + 1) + 1 - 28:28 * (i + 1) + 1] * 255

  scale_percent = 350  # percent of original size
  width = int(im_data.shape[1] * scale_percent / 100)
  height = int(im_data.shape[0] * scale_percent / 100)
  dim = (width, height)

  # resize image
  resized = cv2.resize(im_data, dim, interpolation=cv2.INTER_AREA)
  # Se utilizar o Jupyter notebook utilizar a linha abaixo
  # cv2.imshow("a",resized)
  cv2.imshow("teste", resized)
  if save == True:
    cv2.imwrite(f'num {dataset.iloc[dataset_position][0]} - dt_pos{dataset_position}.jpg', resized)


def teste_acertividade(dataset, a1):
  cont_acert = 0
  for i in range(0, len(dataset)):
    # local_image_array = list(test_dataset.iloc[i,1:65])
    num_real = dataset.iloc[i, 0]
    # print(len(local_image_array))
    num_rede = digit_recog(a1, training_instance=i, dataset=dataset)[0]
    #print(f'Núm. real: {num_real}, núm rede: {num_rede}')
    if num_rede != np.nan:
      if (num_real == num_rede):
        cont_acert += 1

  return 100 * cont_acert / len(dataset)


if __name__=='__main__':
  main()