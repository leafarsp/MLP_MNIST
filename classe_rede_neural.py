import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

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
        self.id = 0
        self.l = list()
        self.weights_initialized = False
        self.fitness = 0
        for i in range(0, L):
            self.l.append(layer(m[i + 1], m[i]))

    def get_weights_connected_ahead(self, j, l):
        wlLkj = np.zeros(self.m[l + 2])
        for k in range(0, self.m[l + 2]):
            wlLkj[k] = self.l[l + 1].w[k][j]
        return wlLkj

    def initialize_weights_random(self, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        weight_limit = 10.
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

        for l in range(0, self.L):
            df[l + 1][0:self.m[l] + 1] = np.transpose(self.l[l].w)

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

    def d_func_ativacao(self, a, b, v):
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

    def set_fitness(self,fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

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
                         momentum, err_min):
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
        a1.initialize_weights_random(random_seed=rnd_seed)

    # Vetor de pesos para plotar gráficos de evolução deles.
    a1plt = list()
    acert = list()

    Eav = np.zeros(n_epoch)
    # início do treinamento
    start_time_epoch = dt.datetime.now()
    for ne in range(0, n_epoch):
        dataset_shufle = dataset.sample(frac=1, random_state=rnd_seed, axis=0)
        e_epoch = 0

        for ni in range(0, n_inst):
            n = ni + ne * (n_inst)
            if n >= (N - 1):
                break
            x = list(dataset_shufle.iloc[ni, 1:(a1.m[0] + 1)])
            output_value = int(dataset_shufle.iloc[ni, 0])
            # d = [dataset_shufle.iloc[ni, 0]]
            d = output_layer_activation(output_value=output_value, num_classes=num_classes)
            a1.forward_propagation(x=x)
            a1.backward_propagation(x=x, d=d, alpha=alpha[n], eta=eta[n])

            if n >= step_plot:
                if n % step_plot == 0:
                    acert.append(teste_acertividade(test_dataset, int(num_classes), a1))
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

def output_layer_activation(output_value, num_classes):
    d = np.ones(num_classes) * -1
    # num = dataset_shufle.iloc[ni, 0]
    d[output_value] = 1.
    return d

def teste_acertividade(test_dataset, num_classes, neural_network):
    cont_acert = 0
    for i in range(0, len(test_dataset)):

        num_real = test_dataset.iloc[i, 0]
        x = list(test_dataset.iloc[i, 1:])

        y = neural_network.forward_propagation(x)
        num_rede = neural_network.get_output_class()

        if num_rede != np.nan:
            if (num_real == num_rede):
                cont_acert += 1

    return 100 * cont_acert / len(test_dataset)

def train_genetic(rede, num_classes, rnd_seed, dataset, test_dataset, num_individuos, step_plot, err_min, target_fitness, mut_prob):

    population = list()

    initialize_population(population, num_individuos, rede, rnd_seed)
    best_ind = get_best_ind(population, 1)

    count_generations = 0
    watchdog = 0

    while(best_ind.fitness < target_fitness and count_generations < 100):
        count_generations += 1

        population_play(dataset, test_dataset, num_classes, population, num_individuos, rede, rnd_seed)

        # Crossover e seleção K tornament
        k = 5
        mutation_probability = 0.05
        elitism = 2

        next_gen = list()
        apply_elitism(population, next_gen, elitism)
        best_ind = get_best_ind(population,1)
        best_fitness = best_ind.fitness

        while((len(population) > elitism) and watchdog < 1000):
            watchdog += 1

            parent1, parent2 = k_tournament(population=population,k=5)

            print(f'Parent1 id: {parent1.id}, Parent1 fitness: {parent1.fitness} '
                  f'Parent2 id: {parent2.id}, Parent2 fitness: {parent2.fitness}')

            kids = crossover(parent1, parent2, mut_prob)

            #mutate(kids,mutation_probability)

            for kid in kids:
                next_gen.append(kid)

            remove_individual(population, [parent1.id, parent2.id])
        population = next_gen

    if watchdog > 1000:
        print('Exit by watchdog.')
    elif count_generations > 100:
        print('Exit by end of generations.')
    else:
        pass
        print(f'Best individual: {best_ind.id}, fitness:{best_ind.id}')
        # salvar indivídio
    print('End of Training')

def population_play(dataset, test_dataset, num_classes, population, num_individuos, rede, rnd_seed):
    n_inst = len(dataset.index)
    for nind in range(0, num_individuos):
        dataset_shufle = dataset.sample(frac=1, random_state=rnd_seed, axis=0)
        for ni in range(0, n_inst):
            x = list(dataset_shufle.iloc[ni, 1:(rede.m[0] + 1)])
            output_value = int(dataset_shufle.iloc[ni, 0])
            # d = [dataset_shufle.iloc[ni, 0]]
            d = output_layer_activation(output_value=output_value, num_classes=num_classes)
            population[nind].forward_propagation(x=x)
            # print(f'individuo {nind}, Y={population[nind].l[rede.L-1].y}')
            acert = teste_acertividade(test_dataset, int(num_classes), population[nind])

            population[nind].set_fitness(acert)

def initialize_population(population, num_individuos, rede, rnd_seed):
    for ni in range(0, num_individuos):
        population.append(rede_neural(rede.L, rede.m, rede.a, rede.b))
        population[-1].initialize_weights_random(ni + rnd_seed)
        population[-1].id = ni

def crossover(parent1, parent2,prob_mut):

    son1 = rede_neural(parent1.L, parent1.m, parent1.a, parent1.b)
    son2 = rede_neural(parent2.L, parent2.m, parent2.a, parent2.b)

    for l in range(0,parent1.L):
        for j in range(0,parent1.m[l+1]):
            for w in range(0,parent1.m[l]+1):

                prob = np.random.randint(2)
                weight_mult1 = get_weight_multiplier(prob_mut)
                weight_mult2 = get_weight_multiplier(prob_mut)

                if prob==0:
                    son1.l[l].w[w] = parent1.l[l].w[w] * weight_mult1
                    son2.l[l].w[w] = parent2.l[l].w[w] * weight_mult2
                else:
                    son1.l[l].w[w] = parent2.l[l].w[w] * weight_mult1
                    son2.l[l].w[w] = parent1.l[l].w[w] * weight_mult2

    return [son1, son2]

def mutate(inds,prob):
    pass

def get_weight_multiplier(mutation_prob):
    weight_mult1 = 1.
    try:
        #val_prob_mut1 = np.random.choice([0,1], p=[1. - mutation_prob, mutation_prob])
        val_prob_mut1 = np.random.choice(2, 1, p=[0.5, 0.5])[0]
        print(val_prob_mut1)
    except:
        print(f'Erro')
    if val_prob_mut1 == 1:
        weight_mult1 = 2.
    return weight_mult1


def get_best_ind(population, position):
    # Não está funcionando ainda, apenas para poder testar as outras funções
    return population[position]


def apply_elitism(population, next_gen, elitism):
    # Não está funcionando ainda, apenas para poder testar as outras funções
    for i in range(0,elitism):
        next_gen.append(get_best_ind(population=population,  position=i))


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

def clone_ind(ind1,ind2):
    pass

def k_tournament(population, k, rnd_seed=None):
    if rnd_seed is not None:
        np.random.seed(rnd_seed)
    fighters = list()

    len_population = len(population)
    fighters.append(np.random.randint(len_population))

    #escolhe aleatóriamente 5 indivídios
    if k > len(population):
        k = len(population)

    while(len(fighters) < k):
        ind = np.random.randint(len_population)
        if ind not in fighters:
            fighters.append(ind)
    parents_ids = [0,0]

    fighters_sorted = [0] * k

    fitness_ant = -999999
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



    return get_ind(population, parents_ids)



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
