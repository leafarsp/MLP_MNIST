import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import classe_rede_neural as nnc
import cv2 as cv2
import time

def main():
  L = 2
  m = [2, 4, 2]
  a = [0.9, 0.9]
  b = [0.5, 0.5]
  # b = [2 / 3, 2 / 3, 2 / 3]
  a1 = nnc.rede_neural(L, m, a, b)

  # a1 = nnc.load_neural_network('Xor_Genetic.xlsx')

  # a1.initialize_weights_random(rnd_seed)

  rnd_seed = 10
  num_classes = 2

  data = {'y': [0, 1, 1, 0], 'x1': [-1, -1, 1, 1], 'x2': [-1, 1, -1, 1]}

  dataset = pd.DataFrame(data=data)
  num_inst = len(dataset.index)

  n_inst = len(dataset.index)

  test_dataset = dataset

  num_individuos = 100
  generations = 2000
  step_plot = 10
  err_min = 0.1
  target_fitness = 0.95
  mut_prob = 0.4
  mutation_multiplyer = 1.
  weight_limit = .2
  elitism = 1
  k = 3
  dataset_division = 1
  # a1.initialize_weights_random()

  population = None
  a1, best_fitness_plt, fitness_list, count_generations, population = nnc.train_genetic(
    rede=a1,
    num_classes=num_classes,
    rnd_seed=rnd_seed,
    dataset=dataset,
    test_dataset=test_dataset,
    num_individuos=num_individuos,
    generations=generations,
    step_plot=step_plot,
    err_min=err_min,
    target_fitness=target_fitness,
    mut_prob=mut_prob,
    weight_limit=weight_limit,
    mutation_multiplyer=mutation_multiplyer,
    elitism=elitism,
    k_tournament_fighters=k,
    dataset_division=dataset_division,
    population=population,
    processor='GPU')

  fitness_list.to_excel('fitness_list_XOR.xlsx')

  a1.save_neural_network("Xor_Genetic.xlsx")

  nnc.teste_neural_network(dataset, a1)

  plt.plot(best_fitness_plt[0:-3])

  # err = nnc.calculate_err_epoch(dataset,a1,a1.output_layer_activation)
  # print(f'erro:{err}')

  acertividade = nnc.teste_acertividade(test_dataset,num_classes,a1)
  #a1.save_neural_network('neural_network2.xlsx')

  buttonPressed = False
  while not buttonPressed:
    buttonPressed = plt.waitforbuttonpress()

  plt_retas(a1, dataset, n_inst)
# def output_layer_activation(output_value):
#   d = np.ones(10) * -1
#   #num = dataset_shufle.iloc[ni, 0]
#   d[output_value] = 1
#   return d
def plt_retas(rede, dataset, num_inst):
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


# def teste_acertividade(dataset, a1):
#   cont_acert = 0
#   for i in range(0, len(dataset)):
#     # local_image_array = list(test_dataset.iloc[i,1:65])
#     num_real = dataset.iloc[i, 0]
#     # print(len(local_image_array))
#     num_rede = digit_recog(a1, training_instance=i, dataset=dataset)[0]
#     print(f'Núm. real: {num_real}, núm rede: {num_rede}')
#     if num_rede != np.nan:
#       if (num_real == num_rede):
#         cont_acert += 1
#
#   return 100 * cont_acert / len(dataset)


if __name__=='__main__':
  main()