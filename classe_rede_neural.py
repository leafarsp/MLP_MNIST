import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import logging
import threading
import time
import sys
from numba import jit, cuda


class layer():
    def __init__(self, m, m_ant):
        self.w = np.ones((m, m_ant + 1))
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
        self.id = 0.
        self.uniqueId = self.__hash__()
        self.l = list()
        self.weights_initialized = False
        self.fitness = 0
        self.acertividade = 0
        self.generation = 0
        self.flag_test_acertividade = False
        for i in range(0, L):
            self.l.append(layer(m[i + 1], m[i]))

    def get_weights_connected_ahead(self, j, l):
        wlLkj = np.zeros(self.m[l + 2])
        for k in range(0, self.m[l + 2]):
            wlLkj[k] = self.l[l + 1].w[k][j]
        return wlLkj

    def initialize_weights_random(self, weight_limit = 10., random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        # weight_limit = 10.
        for l in range(0, self.L):
            self.l[l].w = np.random.rand(self.m[l + 1], self.m[l] + 1) * 2.* (weight_limit) - weight_limit
            # Inicializa o Bias como zero
            for j in range(0, self.m[l + 1]):
                self.l[l].w[j][-1] = 0
        self.weights_initialized = True

    def save_neural_network(self, filename='neural_network.xlsx'):
        max_layer = np.max(self.m)

        data = np.zeros((max_layer + 1, np.sum(self.m[1:])))
        data[:] = np.nan
        arrays = np.zeros((2, np.sum(self.m[1:])))

        end_array = 0
        start_array = 0
        for l in range(0, self.L):

            if l == 0:
                start_array = 0
                end_array = start_array + self.m[l + 1]
            else:
                start_array += self.m[l]
                end_array += self.m[l + 1]

            arrays[0][start_array:end_array] = int(l + 1)
            arrays[1][start_array:end_array] = np.arange(0, self.m[l + 1])

        tuples = list(zip(*arrays))

        columns = pd.MultiIndex.from_tuples(tuples, names=['Layer:', 'Neuron:'])
        df = pd.DataFrame(data=data, columns=columns)
        #print(df)
        for l in range(0, self.L):
            for n in range(0, self.m[l+1]):
                temp_l = np.transpose(self.l[l].w[n])
                # temp_l = np.transpose(temp_l)
                # print(f'camada={l}, neurônio={n}')
                # print(temp_l)
                # print(df.loc[0:self.m[l], l+1].loc[:,n])
                # print(df.loc[0:self.m[l] + 1, l + 1])
                df.loc[0:self.m[l], l + 1].loc[:, n] = temp_l
                #df.loc[0:self.m[l] + 1, l + 1] = temp_l

        #exit()
        data2 = np.zeros((len(self.m), 4))
        data2[:] = np.nan
        df2 = pd.DataFrame(data=data2, columns=['L', 'm', 'a', 'b'])
        df2['L'][0] = self.L
        df2['m'][0:len(self.m)] = self.m
        df2['a'][0:len(self.m) - 1] = self.a
        df2['b'][0:len(self.m) - 1] = self.b

        with pd.ExcelWriter(filename) as writer:
            df.to_excel(writer, sheet_name='weights')
            df2.to_excel(writer, sheet_name='params')

    def activation_func(self, a, b, v):
        # return 1/(1+ np.exp(-a * v))
        return a * np.tanh(b * v)

    # # function optimized to run on gpu
    @jit(target_backend='cuda')
    def activation_func_GPU(self, a, b, v):
        # return 1/(1+ np.exp(-a * v))
        return a * np.tanh(b * v)

    def d_func_ativacao(self, a, b, v):
        # return (a * np.exp(-a * v)) / ((1 + np.exp(-a * v))**2)
        return a * b * (1 - np.tanh(b * v) ** 2)

    # # function optimized to run on gpu
    @jit(target_backend='cuda')
    def d_func_ativacao_GPU(self, a, b, v):
        # return (a * np.exp(-a * v)) / ((1 + np.exp(-a * v))**2)
        return a * b * (1 - np.tanh(b * v) ** 2)

    def forward_propagation(self, x):
        if len(x) != self.m[0]:
            print(
                f'Error, input vector has different size from expected. Input size= {len(x)}, Input nodes = {self.m[0]}')
        input = np.append(x, 1)  # acrescenta 1 relativo ao bias
        for l in range(0, self.L):
            for j in range(0, self.m[l + 1]):
                self.l[l].v[j] = np.matmul(np.transpose(self.l[l].w[j]), input)
                self.l[l].y[j] = self.activation_func(self.a[l], self.b[l], self.l[l].v[j])
            input = np.append(self.l[l].y, 1)
        return self.l[self.L - 1].y

    def forward_propagation_concurrent(self, x):
        if len(x) != self.m[0]:
            print(
                f'Error, input vector has different size from expected. Input size= {len(x)}, Input nodes = {self.m[0]}')
        input = np.append(x, 1)  # acrescenta 1 relativo ao bias
        for l in range(0, self.L):
            thread_list = list()
            for j in range(0, self.m[l + 1]):
                # self.l[l].v[j] = np.matmul(np.transpose(self.l[l].w[j]), input)
                # self.l[l].y[j] = self.activation_func(self.a[l], self.b[l], self.l[l].v[j])
                thread_list.append(threading.Thread(target=self.__compute_neuron__,args=(l, j, input)))
                # thread_list[-1].start()
            for j in range(0, self.m[l + 1]):
                thread_list[j].start()
            for j in range(0, self.m[l + 1]):
                thread_list[j].join()

            input = np.append(self.l[l].y, 1)
        return self.l[self.L - 1].y


    def forward_propagation_GPU(self, x):
        if len(x) != self.m[0]:
            print(
                f'Error, input vector has different size from expected. Input size= {len(x)}, Input nodes = {self.m[0]}')
        input = np.append(x, 1)  # acrescenta 1 relativo ao bias
        for l in range(0, self.L):
            for j in range(0, self.m[l + 1]):
                self.l[l].v[j] = self.__compute_neuron_GPU__(self.l[l].w[j], input)
                # self.l[l].v[j] = np.matmul(np.transpose(self.l[l].w[j]), input)
                self.l[l].y[j] = self.activation_func_GPU(self.a[l], self.b[l], self.l[l].v[j])

            input = np.append(self.l[l].y, 1)
        return self.l[self.L - 1].y

    def __compute_neuron__(self,layer, neuron, input):
        self.l[layer].v[neuron] = np.matmul(np.transpose(self.l[layer].w[neuron]), input)
        self.l[layer].y[neuron] = self.activation_func(self.a[layer], self.b[layer], self.l[layer].v[neuron])

    # # function optimized to run on gpu
    @jit(target_backend='cuda')
    def __compute_neuron_GPU__(self, weight_vector, input_vector):
        temp = np.transpose(weight_vector)
        return np.matmul(np.transpose(weight_vector), input_vector)

    def backward_propagation(self, x, d, alpha, eta):
        if len(d) != self.m[-1]:
            print(
                f'Error, input vector has different size from expected. Input size= {len(x)}, Input nodes = {self.m[-1]}')
        # self.forward_propagation(x)
        output_d = np.append(d, 1)
        for l in range(self.L - 1, -1, -1):
            for j in range(0, self.m[l + 1]):
                # print(f'l={l}, j= {j}')
                if l == (self.L - 1):
                    self.l[l].e[j] = output_d[j] - self.l[l].y[j]
                else:
                    self.l[l].e[j] = np.sum(self.l[l + 1].delta * self.get_weights_connected_ahead(j, l))

                self.l[l].delta[j] = self.l[l].e[j] * self.d_func_ativacao(self.a[l], self.b[l], self.l[l].v[j])
                if l == (0):
                    input = np.append(x, 1)
                else:
                    input = np.append(self.l[l - 1].y, 1)

                w_temp = self.l[l].w[j] + alpha[l] * self.l[l].w_ant[j] + eta[l] * self.l[l].delta[j] * input
                self.l[l].w_ant[j] = np.copy(self.l[l].w[j])
                self.l[l].w[j] = w_temp

    def get_sum_eL(self):
        return np.sum(self.l[-1].e ** 2)


    def calculate_error_inst(self, x, d):
        self.forward_propagation(x)
        return np.sum((d - self.l[self.L - 1].y) ** 2)

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

    def set_acertividade(self, acertividade):
        self.acertividade = acertividade
        self.flag_test_acertividade = True

    def get_acertividade(self):
        return self.acertividade

    def get_flag_teste_acertividade(self):
        return self.flag_test_acertividade


    def get_generation(self):
        return self.generation

    def set_generation(self, generation):
        self.generation = generation

    def get_id(self):
        return self.id
    def set_id(self,id):
        self.id = id

    def get_output_class(self, threshold=0.8):
        num_out = np.nan
        cont_neuronio_ativo = 0
        for j in range(0, self.m[self.L]):
            if (self.l[self.L-1].y[j] > (1 * threshold)):
                num_out = j
                cont_neuronio_ativo += 1
            if (cont_neuronio_ativo > 1):
                num_out = np.nan
                break
        return num_out

    def clone(self):
        clone = rede_neural(self.L, self.m, self.a, self.b)
        clone.set_fitness(self.get_fitness())
        clone.set_generation(self.get_generation())
        clone.set_id(self.get_id())
        clone.acertividade = self.get_acertividade()
        clone.flag_test_acertividade = self.get_flag_teste_acertividade()
        clone.uniqueId = self.uniqueId
        for l in range(0, self.L):
            for j in range(0, self.m[l + 1]):
                for w in range(0, self.m[l] + 1):
                    clone.l[l].w[j][w] = self.l[l].w[j][w]
        return clone

    # # function optimized to run on gpu
    @jit(target_backend='cuda')
    def output_layer_activation_GPU(self, output_value, num_classes):
        d = np.ones(num_classes, dtype=np.float64) * -1
        # num = dataset_shufle.iloc[ni, 0]
        d[output_value] = 1.
        return d

    def output_layer_activation(self, output_value, num_classes):
        d = np.ones(num_classes, dtype=np.float64) * -1
        # num = dataset_shufle.iloc[ni, 0]
        d[output_value] = 1.
        return d






# class train_Neural_Network():
#     def __init__(self):
#         pass

def load_neural_network(neural_network_xlsx):
    df = pd.read_excel(open(neural_network_xlsx, 'rb'),
                       sheet_name='weights')
    df2 = pd.read_excel(open(neural_network_xlsx, 'rb'),
                        sheet_name='params')

    L = int(df2['L'][0])
    m = list(df2['m'][0:L + 1])
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
        # df[l + 1][0:self.m[l] + 1] = np.transpose(self.l[l].w)
        for j in range(0, m[l + 1]):
            # print(np.transpose(df.loc[l + 1][0:m[l] + 1]))
            # print(f'np.shape(np.transpose(df[l + 1][0:m[l] + 1]))={np.shape(np.transpose(df[l + 1][0:m[l] + 3]))}, np.shape(a1.l[l].w) = {np.shape(a1.l[l].w)}\n')
            a1.l[l].w = np.transpose(df[l + 1][0:m[l] + 1].to_numpy())

    a1.weights_initialized = True
    return a1

def train_neural_network(rede, num_classes, rnd_seed, dataset, test_dataset, n_epoch, step_plot, learning_rate,
                         momentum, err_min, weight_limit):
    start_time = dt.datetime.now()

    print(f'Start time: {start_time.year:04d}-{start_time.month:02d}-{start_time.day:02d}'
          f'--{start_time.hour:02d}:{start_time.minute:02d}:{start_time.second:02d}')
    rnd_seed = rnd_seed
    # Base de dados de treinamento
    dataset = dataset
    test_dataset = test_dataset

    # cria rede neural
    a1 = rede

    n_inst = len(dataset.index)
    # parâmetros de treinamento da rede
    n_epoch = n_epoch
    n_inst = len(dataset.index)
    N = n_inst * n_epoch
    step_plot = step_plot

    n_cont = 0

    eta = np.ones((a1.L, N))
    for l in range(0, a1.L):
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

    # Inicializa os pesos com valores aleatórios e o bias como zero
    if a1.weights_initialized == False:
        a1.initialize_weights_random(random_seed=rnd_seed, weight_limit= weight_limit)

    # Vetor de pesos para plotar gráficos de evolução deles.
    a1plt = list()
    acert = list()

    Eav = np.zeros(n_epoch)
    # início do treinamento
    start_time_epoch = dt.datetime.now()
    for ne in range(0, n_epoch):

        dataset_shufle = dataset.sample(frac=1, random_state=rnd_seed, axis=0)
        rnd_seed += 1
        e_epoch = 0

        for ni in range(0, n_inst):
            n = ni + ne * (n_inst)
            if n >= (N - 1):
                break
            x = list(dataset_shufle.iloc[ni, 1:(a1.m[0] + 1)])
            output_value = int(dataset_shufle.iloc[ni, 0])
            # d = [dataset_shufle.iloc[ni, 0]]
            d = a1.output_layer_activation(output_value=output_value, num_classes=num_classes)
            a1.forward_propagation(x=x)
            a1.backward_propagation(x=x, d=d, alpha=alpha[n], eta=eta[n])

            if n >= step_plot:
                if n % step_plot == 0:
                    teste_acertividade(test_dataset, int(num_classes), a1)
                    acert.append(a1.get_acertividade())
                    elapsed_time = dt.datetime.now() - start_time_epoch
                    start_time_epoch = dt.datetime.now()
                    estimated_time_end = start_time_epoch + elapsed_time * (N // step_plot - n_cont)
                    n_cont += 1
                    print(f'Instância {n}/{N}, Época {ne}/{n_epoch}, '
                          f' Acert.: {acert[-1]:.4f}%, eta[L][n]: {eta[n][a1.L-1]:.4f}, dt: {elapsed_time.seconds}s'
                          f' t_end: {estimated_time_end.year:04d}-{estimated_time_end.month:02d}-{estimated_time_end.day:02d}'
                          f'--{estimated_time_end.hour:02d}:{estimated_time_end.minute:02d}:{estimated_time_end.second:02d}')
                    temp_rede = rede_neural(a1.L, a1.m, a1.a, a1.b)
                    for l in range(0, a1.L):
                        temp_rede.l[l].w = np.copy(a1.l[l].w)
                    a1plt.append(temp_rede)
                    # a1.save_neural_network('backup_neural_network.xlsx')

            e_epoch += a1.get_sum_eL()
        Eav[ne] = 1 / (n_inst) * e_epoch

        print(f'Erro Época {ne}/{n_epoch}: {Eav[ne - 1]:.5f}')
        # A linha abaixo calcula a média como escrito no livro, mas
        # não tem muito sentido calcular desse jeito, o erro
        # fica menor se o número de épocas aumenta.
        # Se eu pegar uma rede que foi treinada desse jeito e chegou
        # num erro 0,0001 por exemplo, se eu testá-la novamente
        # com apenas uma época, o erro vai ser maior.
        # Pra mim esse valor deveria ser fíxo, independente do
        # número de épocas. Dessa forma, eu obteria o mesmo erro,
        # seja após 1000 épocas ou após apenas uma.
        # Eav[ne] += Eav[ne] + 1/(2*N) * e_epoch
        if (Eav[ne] < err_min):
            print(f'Erro mínimo: {Eav[ne]}')
            break
    # teste da rede neural

    return a1, a1plt, Eav, n, acert



def calculate_err_epoch(dataset, a1, func_d):
    n_inst = len(dataset.index)
    e_epoch = 0
    # Eav = 0
    for ni in range(0, n_inst):
        d = func_d(dataset.iloc[ni, 0])
        x = list(dataset.iloc[ni, 1:(a1.m[0] + 1)])
        e_epoch += a1.calculate_error_inst(x, d)
    Eav = 1 / (n_inst) * e_epoch
    return Eav

# def output_layer_activation(output_value, num_classes):
#     d = np.ones(num_classes) * -1
#     # num = dataset_shufle.iloc[ni, 0]
#     d[output_value] = 1.
#     return d

def teste_acertividade(test_dataset, num_classes, neural_network):
    cont_acert = 0
    if neural_network.get_flag_teste_acertividade() == False:
        for i in range(0, len(test_dataset)):

            num_real = test_dataset.iloc[i, 0]
            x = list(test_dataset.iloc[i, 1:])

            y = neural_network.forward_propagation(x)

            num_rede = neural_network.get_output_class()

            if num_rede != np.nan:
                if (num_real == num_rede):
                    cont_acert += 1

        result = 100 * cont_acert / len(test_dataset)
        # print(f'Acertividade: {result}')
        neural_network.set_acertividade(result)
    # else:
    #     print(f'ind unique ID: {neural_network.uniqueId}, Acertividade já testada')
    # return result
# TODO: Corrigir algoritmo de divisão do dataset.
def calculate_fitness(test_dataset, rede, num_classes, name=0):
    # logging.info("Thread %s: starting", name)
    n_inst = len(test_dataset.index)
    err_avg = 0
    result = 0
    punishment = 0
    err_class = 0
    err_nan = 0
    err_count_class = [0] * (num_classes+1)
    for i in range(0, len(test_dataset)):

        num_real = test_dataset.iloc[i, 0]
        x = list(test_dataset.iloc[i, 1:])

        y = rede.forward_propagation(x)

        d = rede.output_layer_activation(num_real,num_classes)
        num_rede = rede.get_output_class()


        if num_rede != np.nan:
            if (num_real != num_rede):
                err_class += 1
                err_count_class[int(num_real)] += 1
        else:
            err_nan +=1
            err_count_class[num_classes] += 1
        err = d - y

        err_avg += np.sqrt(np.matmul(err, np.transpose(err))/len(err))

    err_std_dev = np.std(err_count_class)
    err_mean = np.mean(err_count_class)
    err_max = np.max(err_count_class)
    # erro de not a number é menos crítico do que o erro de classe, por isso ele é multiplicado por 0.6
    punishment = (err_class + err_nan*0.1) /n_inst
    punishment *= (1 + (err_mean + err_std_dev)/n_inst)

    #punishment = (err_mean + err_std_dev + err_max)/n_inst


    err_avg = err_avg / n_inst
    b = 1 + 1 / (1 + np.exp(2))
    c = -2
    f = 2
    a = -1
    result = b + a / (1 + np.exp(-f * (err_avg) - c))
    result *= (1- punishment)

    rede.set_fitness(result)
    # logging.info(f'Individual: {neural_network[nind].get_id()}, Generation: {neural_network[nind].neural_network()}, '
    #              f'fitness: {neural_network[nind].get_fitness()}\n')
    # print(f'Individual: {neural_network.get_id()}, Generation: {neural_network.get_generation()}, fitness: {neural_network.get_fitness()}\n')
    #return result
    # logging.info(f'Thread {name}: finished')

def calculate_fitness_GPU(test_dataset, rede, num_classes, name=0):
    # logging.info("Thread %s: starting", name)
    n_inst = len(test_dataset.index)
    err_avg = 0
    result = 0
    punishment = 0
    err_class = 0
    err_nan = 0
    err_count_class = [0] * (num_classes+1)
    for i in range(0, len(test_dataset)):

        num_real = test_dataset.iloc[i, 0]
        x = list(test_dataset.iloc[i, 1:])

        y = rede.forward_propagation_GPU(x)

        d = rede.output_layer_activation(num_real,num_classes)
        num_rede = rede.get_output_class()


        if num_rede != np.nan:
            if (num_real != num_rede):
                err_class += 1
                err_count_class[int(num_real)] += 1
        else:
            err_nan +=1
            err_count_class[num_classes] += 1
        err = d - y

        err_avg += np.sqrt(np.matmul(err, np.transpose(err))/len(err))

    err_std_dev = np.std(err_count_class)
    err_mean = np.mean(err_count_class)
    err_max = np.max(err_count_class)
    # erro de not a number é menos crítico do que o erro de classe, por isso ele é multiplicado por 0.6
    punishment = (err_class + err_nan*0.1) /n_inst
    punishment *= (1 + (err_mean + err_std_dev)/n_inst)

    #punishment = (err_mean + err_std_dev + err_max)/n_inst


    err_avg = err_avg / n_inst
    b = 1 + 1 / (1 + np.exp(2))
    c = -2
    f = 2
    a = -1
    result = b + a / (1 + np.exp(-f * (err_avg) - c))
    result *= (1- punishment)

    rede.set_fitness(result)
    # logging.info(f'Individual: {neural_network[nind].get_id()}, Generation: {neural_network[nind].neural_network()}, '
    #              f'fitness: {neural_network[nind].get_fitness()}\n')
    # print(f'Individual: {neural_network.get_id()}, Generation: {neural_network.get_generation()}, fitness: {neural_network.get_fitness()}\n')
    #return result
    # logging.info(f'Thread {name}: finished')



def teste_neural_network(test_dataset, neural_network):
    cont_acert = 0
    acert = 0
    for i in range(0, len(test_dataset)):
        result = 'Wrong'
        num_real = test_dataset.iloc[i, 0]
        x = list(test_dataset.iloc[i, 1:])

        y = neural_network.forward_propagation(x)
        num_rede = neural_network.get_output_class()

        if num_rede != np.nan:
            if (num_real == num_rede):
                cont_acert += 1
                result = 'OK'

        print(f'input:{x} output: {num_rede}, output desired: {num_real}, result: {result}')
    acert = 100 * cont_acert / len(test_dataset)
    print(f'Acertividade: {acert:.2f}%')
    return acert

# TODO: verificar a reprodutibilidade dos dados, colocando as sementes dos valores aleatórios nos lugares corretos.

def train_genetic(rede, num_classes, rnd_seed, dataset, test_dataset,
                  num_individuos, generations, step_plot, err_min, target_fitness,
                  mut_prob, weight_limit, mutation_multiplyer, elitism, k_tournament_fighters, dataset_division = None, population=None):
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    logging.info(f'Starting training')
    local_rnd_seed = rnd_seed
    count_generations = 0
    watchdog = 0
    best_fitness_plt = np.zeros(generations)

    if population == None:
        population = list()
        initialize_population(population, num_individuos, rede, rnd_seed, weight_limit)
    best_ind = get_best_ind(population, 0)

    acert = 0
    start_time = time.time()
    end_time = time.time()
    elapsed_time = 0
    while(best_ind.fitness < target_fitness and count_generations < generations ):
        elapsed_time = end_time - start_time
        start_time = time.time()
        count_generations += 1
        # print(f'\nGeneration:{count_generations}\n')

        population_play_concurent(dataset, test_dataset, num_classes, population,  rede, local_rnd_seed,
                        count_generations, best_ind, best_ind.get_acertividade(), dataset_division)
        # population_play(dataset, test_dataset, num_classes, population, rede, rnd_seed,
        #                           count_generations, best_ind, best_ind.get_acertividade(), dataset_division)

        print(f'count_generations={count_generations:04d}/{generations:04d}, Best: {best_ind.id:04d},'
              f' best uniqueID: {best_ind.uniqueId} '
              f'Best generation: {best_ind.get_generation():04d}, fitness:{best_ind.fitness:.10f}, '
              f'Acertividade: {best_ind.get_acertividade():.7f}%, generation_time: {elapsed_time:.3f}')
        local_rnd_seed += 1


        fitness_list = get_fitness_list(population)
        # Crossover e seleção K tornament
        # for ind1 in population:
        #     print(f'ind1.uniqueID: {ind1.uniqueId}, ind1.fitness={ind1.get_fitness():.10f}'
        #           f' ind1.id = {ind1.get_id()}')

        best_ind = get_best_ind(population, 0)
        # print(f'\nbest_ind.uniqueID: {best_ind.uniqueId}, best_ind.fitness={best_ind.get_fitness():.10f} '
        #       f'best_ind id: {best_ind.get_id()} Best_ind.flag_acertividade = {best_ind.get_flag_teste_acertividade()}'
        #       f' Acertividade: {best_ind.get_acertividade():.7f}%')



        # # print(f'Best_ind.flag_acertividade = {best_ind.get_flag_teste_acertividade()}')
        # if best_ind.get_flag_teste_acertividade() != True:
        #     print(f'Testando acertividade do melhor indivíduo, ger. {best_ind.get_generation()}, '
        #           f'ind1.uniqueID: {best_ind.uniqueId}, id {best_ind.get_id()}')
        #     # test_dataset_shufle = test_dataset.sample(frac=1, random_state=rnd_seed, axis=0)
        #     # test_dataset_shufle = test_dataset_shufle.iloc[0:int(len(test_dataset.index) / 10)]
        #     teste_acertividade(test_dataset, num_classes, best_ind)
        #     # best_ind.set_acertividade(acert)
        #     print(f'Acertividade: {best_ind.get_acertividade():.7f}%'
        #           f' best_ind.uniqueID: {best_ind.uniqueId}, best id: {best_ind.get_id()}')

        # print(f'best_ind.fitness={best_ind.get_fitness():.10f} best_ind.uniqueID: {best_ind.uniqueId} '
        #       f'best_ind id: {best_ind.get_id()} Best_ind.flag_acertividade = {best_ind.get_flag_teste_acertividade()}'
        #       f' Acertividade: {best_ind.get_acertividade():.7f}%')


        if count_generations < (generations-1):
            next_gen = list()
            apply_elitism(population, next_gen, elitism)



            best_fitness_plt[count_generations - 1] = best_ind.fitness
            watchdog=0
            while (len(population)>elitism):


                parent1, parent2 = k_tournament(population=population,k=k_tournament_fighters)

                #print(f'Parent1 id: {parent1.id}, Parent1 fitness: {parent1.fitness} '
                #      f'Parent2 id: {parent2.id}, Parent2 fitness: {parent2.fitness}')

                kids = crossover(parent1, parent2, mut_prob, mutation_multiplyer)

                #mutate(kids,mutation_probability)

                for kid in kids:
                    if len(next_gen) < num_individuos:
                        next_gen.append(kid)
                        next_gen[-1].id = len(next_gen)-1
                        next_gen[-1].set_generation(count_generations)

                remove_individual(population, [parent1.id, parent2.id])
            population = next_gen
        # if count_generations % 5 == 0 or count_generations == 1:




        end_time = time.time()

    if watchdog > 1000:
        print('Exit by watchdog.')
    elif count_generations >= generations:
        print('Exit by end of generations.')

    print(f'Best individual: {best_ind.id}, fitness:{best_ind.fitness}')
    print(f'Acertividade: {acert}%')
        # salvar indivídio
    logging.info(f'End of Training')
    best_fitness_plt[-1] = best_fitness_plt[-2]

    return best_ind, best_fitness_plt, fitness_list, count_generations, population

def train_genetic_GPU(rede, num_classes, rnd_seed, dataset, test_dataset,
                  num_individuos, generations, step_plot, err_min, target_fitness,
                  mut_prob, weight_limit, mutation_multiplyer, elitism, k_tournament_fighters, dataset_division = None, population=None):
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    logging.info(f'Starting training')
    local_rnd_seed = rnd_seed
    count_generations = 0
    watchdog = 0
    best_fitness_plt = np.zeros(generations)

    if population == None:
        population = list()
        initialize_population(population, num_individuos, rede, rnd_seed, weight_limit)
    best_ind = get_best_ind(population, 0)

    acert = 0
    start_time = time.time()
    end_time = time.time()
    elapsed_time = 0
    while(best_ind.fitness < target_fitness and count_generations < generations ):
        elapsed_time = end_time - start_time
        start_time = time.time()
        count_generations += 1
        # print(f'\nGeneration:{count_generations}\n')

        population_play_concurent_GPU(dataset, test_dataset, num_classes, population,  rede, local_rnd_seed,
                        count_generations, best_ind, best_ind.get_acertividade(), dataset_division)
        # population_play(dataset, test_dataset, num_classes, population, rede, rnd_seed,
        #                           count_generations, best_ind, best_ind.get_acertividade(), dataset_division)

        print(f'count_generations={count_generations:04d}/{generations:04d}, Best: {best_ind.id:04d},'
              f' best uniqueID: {best_ind.uniqueId} '
              f'Best generation: {best_ind.get_generation():04d}, fitness:{best_ind.fitness:.10f}, '
              f'Acertividade: {best_ind.get_acertividade():.7f}%, generation_time: {elapsed_time:.3f}')
        local_rnd_seed += 1


        fitness_list = get_fitness_list(population)
        # Crossover e seleção K tornament
        # for ind1 in population:
        #     print(f'ind1.uniqueID: {ind1.uniqueId}, ind1.fitness={ind1.get_fitness():.10f}'
        #           f' ind1.id = {ind1.get_id()}')

        best_ind = get_best_ind(population, 0)
        # print(f'\nbest_ind.uniqueID: {best_ind.uniqueId}, best_ind.fitness={best_ind.get_fitness():.10f} '
        #       f'best_ind id: {best_ind.get_id()} Best_ind.flag_acertividade = {best_ind.get_flag_teste_acertividade()}'
        #       f' Acertividade: {best_ind.get_acertividade():.7f}%')



        # # print(f'Best_ind.flag_acertividade = {best_ind.get_flag_teste_acertividade()}')
        # if best_ind.get_flag_teste_acertividade() != True:
        #     print(f'Testando acertividade do melhor indivíduo, ger. {best_ind.get_generation()}, '
        #           f'ind1.uniqueID: {best_ind.uniqueId}, id {best_ind.get_id()}')
        #     # test_dataset_shufle = test_dataset.sample(frac=1, random_state=rnd_seed, axis=0)
        #     # test_dataset_shufle = test_dataset_shufle.iloc[0:int(len(test_dataset.index) / 10)]
        #     teste_acertividade(test_dataset, num_classes, best_ind)
        #     # best_ind.set_acertividade(acert)
        #     print(f'Acertividade: {best_ind.get_acertividade():.7f}%'
        #           f' best_ind.uniqueID: {best_ind.uniqueId}, best id: {best_ind.get_id()}')

        # print(f'best_ind.fitness={best_ind.get_fitness():.10f} best_ind.uniqueID: {best_ind.uniqueId} '
        #       f'best_ind id: {best_ind.get_id()} Best_ind.flag_acertividade = {best_ind.get_flag_teste_acertividade()}'
        #       f' Acertividade: {best_ind.get_acertividade():.7f}%')


        if count_generations < (generations-1):
            next_gen = list()
            apply_elitism(population, next_gen, elitism)



            best_fitness_plt[count_generations - 1] = best_ind.fitness
            watchdog=0
            while (len(population)>elitism):


                parent1, parent2 = k_tournament(population=population,k=k_tournament_fighters)

                #print(f'Parent1 id: {parent1.id}, Parent1 fitness: {parent1.fitness} '
                #      f'Parent2 id: {parent2.id}, Parent2 fitness: {parent2.fitness}')

                kids = crossover(parent1, parent2, mut_prob, mutation_multiplyer)

                #mutate(kids,mutation_probability)

                for kid in kids:
                    if len(next_gen) < num_individuos:
                        next_gen.append(kid)
                        next_gen[-1].id = len(next_gen)-1
                        next_gen[-1].set_generation(count_generations)

                remove_individual(population, [parent1.id, parent2.id])
            population = next_gen
        # if count_generations % 5 == 0 or count_generations == 1:




        end_time = time.time()

    if watchdog > 1000:
        print('Exit by watchdog.')
    elif count_generations >= generations:
        print('Exit by end of generations.')

    print(f'Best individual: {best_ind.id}, fitness:{best_ind.fitness}')
    print(f'Acertividade: {acert}%')
        # salvar indivídio
    logging.info(f'End of Training')
    best_fitness_plt[-1] = best_fitness_plt[-2]

    return best_ind, best_fitness_plt, fitness_list, count_generations, population

def population_play_concurent(dataset, test_dataset, num_classes, population,  rede, rnd_seed, generation,
                              best_ind,acertividade, dataset_division = 1):
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    best_clone = best_ind.clone()

    num_individuos = len(population)
    n_inst = len(dataset.index)



    max_inst_by_ind = int(n_inst / (num_individuos -1))

    if dataset_division > (num_individuos - 1):
        dataset_division = int(num_individuos - 1)

    inst_by_ind = int(n_inst / dataset_division)
    dist = int(inst_by_ind - max_inst_by_ind)

    thread_list = list()

    dataset_shufle = dataset.sample(frac=1, random_state=rnd_seed, axis=0)
    # dataset_shufle.reset_index(inplace=True)
    # dataset_shufle.drop(columns=['index'], axis=1, inplace=True)

    for nind in range(0, num_individuos):
        #inst_inicial = nind * dn
        # inst_final = inst_inicial + dn - 1
        inst_inicial = int((n_inst-dist)/(num_individuos-1)) * nind
        inst_final =  int(inst_inicial + inst_by_ind -1)
        if inst_final >= n_inst:
            inst_inicial -= inst_final - n_inst +1
            inst_final = n_inst-1


        # print(f'inst_inicial:{inst_inicial}, inst_final:{inst_final}')
        # inst_inicial = nind * dn
        # inst_final = inst_inicial + dn - 1

        # thread_list.append(threading.Thread(target=calculate_fitness,
        #                                     args=(dataset_shufle[0:], population[nind], int(num_classes), nind)))
        thread_list.append(threading.Thread(target=calculate_fitness,
                                            args=(dataset_shufle.iloc[inst_inicial:inst_final+1], population[nind],
                                                  num_classes, nind)))
    best_acert_thread = threading.Thread(target=teste_acertividade, args=(test_dataset, int(num_classes), best_clone))
    best_acert_thread.start()
    for nind in range(0, num_individuos):
        thread_list[nind].start()

    for nind in range(0,num_individuos):
        thread_list[nind].join()
    best_acert_thread.join()
    # print(f'best_clone.get_acertividade() = {best_clone.get_acertividade()}')
    best_ind.set_acertividade(best_clone.get_acertividade())
    # print(f'best_ind.get_acertividade() = {best_ind.get_acertividade()}')
    # for nind in range(0, num_individuos):
    #     logging.info(f'Individual: {population[nind].get_id()}, Generation: {population[nind].get_generation()}, '
    #           f'fitness: {population[nind].get_fitness()}\n')

def population_play_concurent_GPU(dataset, test_dataset, num_classes, population,  rede, rnd_seed, generation,
                              best_ind,acertividade, dataset_division = 1):
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    best_clone = best_ind.clone()

    num_individuos = len(population)
    n_inst = len(dataset.index)



    max_inst_by_ind = int(n_inst / (num_individuos -1))

    if dataset_division > (num_individuos - 1):
        dataset_division = int(num_individuos - 1)

    inst_by_ind = int(n_inst / dataset_division)
    dist = int(inst_by_ind - max_inst_by_ind)

    thread_list = list()

    dataset_shufle = dataset.sample(frac=1, random_state=rnd_seed, axis=0)

    for nind in range(0, num_individuos):
        #inst_inicial = nind * dn
        # inst_final = inst_inicial + dn - 1
        inst_inicial = int((n_inst-dist)/(num_individuos-1)) * nind
        inst_final =  int(inst_inicial + inst_by_ind -1)
        if inst_final >= n_inst:
            inst_inicial -= inst_final - n_inst +1
            inst_final = n_inst-1

        # inst_inicial = nind * dn
        # inst_final = inst_inicial + dn - 1

        # thread_list.append(threading.Thread(target=calculate_fitness,
        #                                     args=(dataset_shufle[0:], population[nind], int(num_classes), nind)))
        thread_list.append(threading.Thread(target=calculate_fitness_GPU,
                                            args=(dataset_shufle.iloc[inst_inicial:inst_final+1], population[nind],
                                                  num_classes, nind)))
    best_acert_thread = threading.Thread(target=teste_acertividade, args=(test_dataset, int(num_classes), best_clone))
    best_acert_thread.start()
    for nind in range(0, num_individuos):
        thread_list[nind].start()

    for nind in range(0,num_individuos):
        thread_list[nind].join()
    best_acert_thread.join()
    # print(f'best_clone.get_acertividade() = {best_clone.get_acertividade()}')
    best_ind.set_acertividade(best_clone.get_acertividade())
    # print(f'best_ind.get_acertividade() = {best_ind.get_acertividade()}')
    # for nind in range(0, num_individuos):
    #     logging.info(f'Individual: {population[nind].get_id()}, Generation: {population[nind].get_generation()}, '
    #           f'fitness: {population[nind].get_fitness()}\n')

def population_play(dataset, test_dataset, num_classes, population, rede, rnd_seed, generation, best_ind,
                    acertividade, dataset_division = 1):
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    num_individuos = len(population)
    n_inst = len(dataset.index)
    b = 1 + 1 / (1 + np.exp(2))
    c = -2
    f = 2
    a = -1

    dn = int(n_inst / num_individuos)

    max_inst_by_ind = int(n_inst / (num_individuos - 1))
    if dataset_division > (num_individuos - 1):
        dataset_division = int(num_individuos - 1)

    inst_by_ind = int(n_inst / dataset_division)
    dist = int(inst_by_ind - max_inst_by_ind)

    for nind in range(0, num_individuos):
        dataset_shufle = dataset.sample(frac=1, random_state=rnd_seed, axis=0)

        # inst_inicial = nind * dn
        # inst_final = inst_inicial + dn - 1

        inst_inicial = int((n_inst - dist) / (num_individuos - 1)) * nind
        inst_final = int(inst_inicial + inst_by_ind - 1)
        if inst_final >= n_inst:
            inst_inicial -= inst_final - n_inst +1
            inst_final = n_inst-1
        # print(f'Gen: {generation:04d}, Ind:{nind:04d}, range inst {inst_inicial}->{inst_final}/{n_inst}')

        calculate_fitness(dataset_shufle.iloc[inst_inicial:inst_final+1],population[nind],int(num_classes))
        #acert = calculate_fitness(dataset_shufle[inst_inicial:inst_final], population[nind], int(num_classes))


        # print(f'Geração: {generation:04d}, Individuo: {nind:04d}, fitness: {acert:.7f}, '
        #       f'best_ind_generation = {best_ind.get_generation():04d}, best_fitness: '
              # f'{best_ind.fitness:.7f}, best_acertividade: {acertividade:.3f}%')

        # population[nind].set_fitness(acert)
    teste_acertividade(test_dataset, int(num_classes), best_ind)
        # acert = b + a / (1 + np.exp(-f * err_avg - c))
        # population[nind].set_fitness(acert)
    # for nind in range(0, num_individuos):
    #     logging.info(f'Individual: {population[nind].get_id()}, Generation: {population[nind].get_generation()}, '
    #           f'fitness: {population[nind].get_fitness()}\n')

def initialize_population(population, num_individuos, rede, rnd_seed, weight_limit):

    for ni in range(0, num_individuos):
        if ni == 0:
            population.append(rede)
            if rede.weights_initialized == False:
                population[-1].initialize_weights_random(random_seed=ni + rnd_seed, weight_limit=weight_limit)
        else:
            population.append(rede_neural(rede.L, rede.m, rede.a, rede.b))
            population[-1].initialize_weights_random(random_seed=ni + rnd_seed, weight_limit= weight_limit)
        population[-1].id = ni

def crossover(parent1, parent2,prob_mut, mutation_multiplyer):
    son1 = rede_neural(parent1.L, parent1.m, parent1.a, parent1.b)
    son2 = rede_neural(parent2.L, parent2.m, parent2.a, parent2.b)

    for l in range(0,parent1.L):

        for j in range(0,parent1.m[l+1]):
            prob = np.random.randint(2)
            weight_mult1 = get_weight_multiplier(prob_mut, mutation_multiplyer)
            weight_mult2 = get_weight_multiplier(prob_mut, mutation_multiplyer)
            for w in range(0,parent1.m[l]+1):
                if prob==0:
                    son1.l[l].w[j][w] = parent1.l[l].w[j][w] + weight_mult1
                    son2.l[l].w[j][w] = parent2.l[l].w[j][w] + weight_mult2
                else:
                    son1.l[l].w[j][w] = parent2.l[l].w[j][w] + weight_mult1
                    son2.l[l].w[j][w] = parent1.l[l].w[j][w] + weight_mult2
    return [son1, son2]

def mutate(inds,prob):
    pass

def get_mutation_permission(probability):
    prob = probability
    prob_precision = 100
    prob_int = int(prob * prob_precision)
    num_al = np.random.randint(prob_precision)
    rng_prob = np.arange(0, prob_int)
    result = False
    if num_al in rng_prob:
        result = True
    return result

def get_weight_multiplier(mutation_prob, mutation_multiplyer):
    weight_mult1 = 0.
    #signal = -1 + 2 * np.random.randint(2)
    n_rand = -mutation_multiplyer + 2 * mutation_multiplyer * np.random.rand()
    if get_mutation_permission(mutation_prob):
        #weight_mult1 = mutation_multiplyer * n_rand
        weight_mult1 = n_rand
    return weight_mult1


def get_best_ind(population, rank):
    fitness_list = get_fitness_list(population)
    position = fitness_list.iloc[rank]['position']
    best_ind = get_ind(population,[position])
    return best_ind[0]

def get_fitness_list(population):
    dataset = pd.DataFrame(data=np.zeros((len(population),3)),columns=['id','position','fitness'])
    for i in range(0,len(population)):
        dataset.loc[i]=[population[i].id,i,population[i].fitness]
    dataset.sort_values(by=['fitness'],ascending=False, inplace=True)
    dataset.reset_index(inplace=True)
    dataset.drop(columns=['index'],axis=1,inplace=True)
    return dataset

def apply_elitism(population, next_gen, elitism):

    for i in range(0,elitism):
        best_ind = get_best_ind(population=population,  rank=i)
        best_ind_clone = best_ind.clone()
        best_ind_clone.id = i
        next_gen.append(best_ind_clone)




def get_ind(population, id_list):

    local_list = list()
    if type(id_list) is not list:
        local_list.append(id_list)
    else:
        local_list = id_list
    inds = list()
    for ind in population:
        if ind.id in local_list:
            inds.append(ind)
    return inds



def k_tournament(population, k, rnd_seed=None):
    # próxima atualização: fazer utilizando a função .iloc do pandas
    # agora tá funcionando, mas utilizando a função do pandas pode ficar com menos linhas
    if rnd_seed is not None:
        np.random.seed(rnd_seed)
    fighters = list()

    len_population = len(population)
    fighters.append(np.random.randint(len_population))

    #escolhe aleatóriamente k indivídios
    if k > len(population):
        k = len(population)

    while(len(fighters) < k):
        ind = np.random.randint(len_population)
        if ind not in fighters:
            fighters.append(ind)
    parents_ids = [0,0]

    fighters_sorted = [0] * k

    fitness_ant = -sys.maxsize
    parents_ids[0] = None
    # definição do pai
    pai = 0
    for i in range(0, k):
        if population[fighters[i]].fitness > fitness_ant:
            parents_ids[0] = population[fighters[i]].id
            fitness_ant = population[fighters[i]].fitness
            pai = i
    # definição da mãe
    fitness_ant = -999999
    parents_ids[1] = None
    for i in range(0, k):
        if i != pai:
            if population[fighters[i]].fitness > fitness_ant:
                parents_ids[1] = population[fighters[i]].id
                fitness_ant = population[fighters[i]].fitness

    parent1, parent2 = get_ind(population, parents_ids)
    return parent1, parent2

def sort_population(population):
    pass

def remove_individual(population, id_list):
    local_list = list()
    if type(id_list) is not list:
        local_list.append(id_list)
    else:
        local_list = id_list

    for ind_remove in local_list:
        for ind in population:
            if ind.id == ind_remove:
                population.remove(ind)
                break

def save_population(population,filename):
    rank = get_fitness_list(population)
    for i in range(0,len(population)):
        id = int(rank.loc[i]['id'])
        print(f'Saving population {i:04d}/{len(population):04d}, id={id}, fitness={population[id].get_fitness():.7f}')
        population[id].save_neural_network(filename=f'{filename}_{i:04d}.xlsx')

def load_population(filename, num_individuos, rede):
    population = list()

    for i in range(0, num_individuos):
        print(f'Loading population {i:04d}/{num_individuos:04d}')
        population.append(load_neural_network(f'{filename}_{i:04d}.xlsx'))
        population[-1].id = i

    return population