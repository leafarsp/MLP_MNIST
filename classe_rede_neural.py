import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class layer():
    def __init__(self, m, m_ant):
        self.w = np.ones((m, m_ant+1))
        self.w_ant = np.zeros((m, m_ant + 1))
        self.y = np.ones(m)
        self.d = np.ones(m)
        self.v = np.ones(m)
        self.delta = np.ones(m)
        self.e = np.ones(m)



class rede_neural():

    def __init__(self, L, m, a, b):


        self.L = L
        self.m = m
        self.a = a
        self.b = b
        self.l = list()
        for i in range(0, L):
            self.l.append(layer(m[i + 1], m[i]))

    def get_weights_connected_ahead(self, j, l):
      wlLkj = np.zeros(self.m[l + 2])
      for k in range(0, self.m[l + 2]):
        wlLkj[k] = self.l[l+1].w[k][j]
      return wlLkj

    def save_neural_network(self, filename='neural_network.xlsx'):
        max_layer = np.max(self.m)

        data = np.zeros((max_layer+1,np.sum(self.m[1:])))
        data[:] = np.nan
        arrays = np.zeros((2,np.sum(self.m[1:])))

        end_array = 0
        start_array = 0
        for l in range(0, self.L):

            if l==0:
                start_array = 0
                end_array = start_array + self.m[l + 1]
            else:
                start_array += self.m[l]
                end_array += self.m[l + 1]

            arrays[0][start_array:end_array] = int(l + 1)
            arrays[1][start_array:end_array] = np.arange(0,self.m[l+1])

        tuples = list(zip(*arrays))


        columns = pd.MultiIndex.from_tuples(tuples, names=['Layer:', 'Neuron:'])
        df = pd.DataFrame(data=data, columns=columns)

        for l in range(0,self.L):
            df[l+1][0:self.m[l]+1] = np.transpose(self.l[l].w)





        data2 = np.zeros((len(self.m), 4))
        data2[:]=np.nan
        df2 = pd.DataFrame(data = data2,columns=['L','m','a','b'])
        df2['L'][0] = self.L
        df2['m'][0:len(self.m)] = self.m
        df2['a'][0:len(self.m)-1] = self.a
        df2['b'][0:len(self.m) - 1] = self.b


        with pd.ExcelWriter(filename) as writer:
            df.to_excel(writer, sheet_name='weights')
            df2.to_excel(writer, sheet_name='params')





    def activation_func(self, a, b, v):
      # return 1/(1+ np.exp(-a * v))
      return a * np.tanh(b * v)

    def d_func_ativacao(self, a, b, v):
      # return (a * np.exp(-a * v)) / ((1 + np.exp(-a * v))**2)
      return a * b * (1 - np.tanh(b * v) ** 2)

    def forward_propagation(self, x):
      if len(x) != self.m[0]:
        print(f'Error, input vector has different size from expected. Input size= {len(x)}, Input nodes = {self.m[0]}')
      input = np.append(x,1) # acrescenta 1 relativo ao bias
      for l in range(0,self.L):
        for j in range(0, self.m[l+1]):
          self.l[l].v[j] = np.matmul(np.transpose(self.l[l].w[j]), input)
          self.l[l].y[j] = self.activation_func(self.a[l], self.b[l], self.l[l].v[j])
        input = np.append(self.l[l].y, 1)
      return self.l[self.L-1].y


    def backward_propagation(self, x, d, alpha, eta):
        if len(d) != self.m[-1]:
            print(
                f'Error, input vector has different size from expected. Input size= {len(x)}, Input nodes = {self.m[-1]}')
        #self.forward_propagation(x)
        output_d = np.append(d,1)
        for l in range(self.L-1, -1, -1):
            for j in range(0, self.m[l + 1]):
                # print(f'l={l}, j= {j}')
                if l == (self.L - 1):
                    self.l[l].e[j] = output_d[j] - self.l[l].y[j]
                else:
                    self.l[l].e[j] = np.sum(self.l[l+1].delta * self.get_weights_connected_ahead(j,l))

                self.l[l].delta[j] = self.l[l].e[j] * self.d_func_ativacao(self.a[l], self.b[l], self.l[l].v[j])
                if l == (0):
                    input = np.append(x,1)
                else:
                    input = np.append(self.l[l-1].y,1)

                w_temp = self.l[l].w[j] + alpha[l] * self.l[l].w_ant[j] + eta[l] * self.l[l].delta[j] * input
                self.l[l].w_ant[j] = np.copy(self.l[l].w[j])
                self.l[l].w[j] = w_temp
    def get_sum_eL(self):
        return np.sum(self.l[-1].e ** 2)

# class train_Neural_Network():
#     def __init__(self):
#         pass

def load_neural_network(neural_network_xlsx):

    df = pd.read_excel(open(neural_network_xlsx, 'rb'),
                       sheet_name='weights')
    df2 = pd.read_excel(open(neural_network_xlsx, 'rb'),
                        sheet_name='params')

    L = int(df2['L'][0])
    m = list(df2['m'][0:L+1])
    a = list(df2['a'][0:L])
    b = list(df2['b'][0:L])

    a1 = rede_neural(L, m, a, b)
    cont_neuron = 0

    # Carrega os pesos




    # faz o array para representar as colunas no excel, igual
    # ao que é feito ao salvar a rede




    arrays = np.zeros((2, np.sum(m[1:])))

    end_array = 0
    start_array = 0
    for l in range(0, L):

        if l == 0:
            start_array = 0
            end_array = start_array + m[l + 1]
        else:
            start_array += m[l]
            end_array += m[l + 1]

        arrays[0][start_array:end_array] = int(l + 1)
        arrays[1][start_array:end_array] = np.arange(0, m[l + 1])

    tuples = list(zip(*arrays))

    columns = pd.MultiIndex.from_tuples(tuples, names=['Layer:', 'Neuron:'])

    data = df.to_numpy()
    data = np.delete(data, 0, 1)
    data = data[2:]
    # print(f'{data[0]}')
    # print(f'{data[-1]}')
    # print(f'np.shape(data)={np.shape(data)}')
    df = pd.DataFrame(data=data, columns=columns)

    for l in range(0, L):
        #df[l + 1][0:self.m[l] + 1] = np.transpose(self.l[l].w)
        for j in range(0, m[l+1]):
            #print(np.transpose(df.loc[l + 1][0:m[l] + 1]))
            #print(f'np.shape(np.transpose(df[l + 1][0:m[l] + 1]))={np.shape(np.transpose(df[l + 1][0:m[l] + 3]))}, np.shape(a1.l[l].w) = {np.shape(a1.l[l].w)}\n')
            a1.l[l].w = np.transpose(df[l + 1][0:m[l] + 1].to_numpy())

    return a1

def train_neural_network(rede, rnd_seed, dataset, test_dataset, n_epoch, step_plot, learning_rate, momentum, err_min, func_d):
  rnd_seed = rnd_seed
  # Base de dados de treinamento
  dataset = dataset
  test_dataset = test_dataset

   # cria rede neural
  a1 = rede



  n_inst = len(dataset.index)
  #parâmetros de treinamento da rede
  n_epoch = n_epoch
  n_inst = len(dataset.index)
  N = n_inst * n_epoch
  step_plot = step_plot




  eta = np.ones((a1.L,N))
  for l in range(0,a1.L):
    eta[l] = list(np.linspace(learning_rate[l], 0., N))

  # for l in range(0,a1.L):
  #   plt.plot(eta[l])
  #   pass
  eta = np.transpose(eta)
  # eta[:, [1, 0]]

# plt.figure()

  alpha = np.ones((a1.L, N))
  for l in range(0, a1.L):
    alpha[l] = list(np.linspace(momentum[l], 0., N))
  # alpha[0] *= 0.000000  # camada de entrada
  # alpha[1] *= 0.000000  # camada oculta 1
  # alpha[2] *= 0.000000  # camada de saída
  # alpha[3] *= 0.000000  # camada de saída
  alpha = np.transpose(alpha)

  # Inicializa os pesos com valores aleatórios
  for l in range(0, a1.L):
    a1.l[l].w = np.random.rand(a1.m[l+1], a1.m[l] + 1) * 1.0 - 0.5
    # Inicializa o Bias como zero
    for j in range(0, a1.m[l+1]):
        a1.l[l].w[j][-1] = 0




  # Vetor de pesos para plotar gráficos de evolução deles.
  a1plt = list()

  Eav = np.zeros(n_epoch)
  #início do treinamento
  for ne in range(0,n_epoch):
    dataset_shufle = dataset.sample(frac=1, random_state=rnd_seed, axis=0)
    e_epoch = 0
    for ni in range(0, n_inst):
      n = ni + ne*(n_inst)
      if n >= (N-1):
        break
      x = list(dataset_shufle.iloc[ni, 1:(a1.m[0]+1)])
      #d = [dataset_shufle.iloc[ni, 0]]
      d = func_d(dataset_shufle.iloc[ni, 0])
      a1.forward_propagation(x=x)
      a1.backward_propagation(x=x, d=d, alpha=alpha[n], eta=eta[n])

      if n >= step_plot:
        if n % step_plot == 0:
          print(f'Instância {n}/{N}, Época {ne}/{n_epoch}, Erro: {Eav[ne-1]}')
          temp_rede = rede_neural(a1.L, a1.m, a1.a, a1.b)
          for l in range(0,a1.L):
            temp_rede.l[l].w = np.copy(a1.l[l].w)
          a1plt.append(temp_rede)

      e_epoch += a1.get_sum_eL()
    Eav[ne] += Eav[ne] + 1/(2*N) * e_epoch
    if (Eav[ne] < err_min):
      print(f'Erro mínimo: {Eav[ne]}')
      break
  #teste da rede neural

  return a1, a1plt, Eav, n