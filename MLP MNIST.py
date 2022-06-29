import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import classe_rede_neural as nnc




def digit_recog(rede, image_array=None, training_instance=0, dataset = None):
    # progagação do sinal forward

    if image_array == None:
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
    cv2.imshow("teste",resized)
    if save == True:
        cv2.imwrite(f'num {dataset.iloc[dataset_position][0]} - dt_pos{dataset_position}.jpg', resized)


def teste_acertividade(dataset, a1):
    cont_acert = 0
    for i in range(0, len(dataset)):
        # local_image_array = list(test_dataset.iloc[i,1:65])
        num_real = dataset.iloc[i, 0]
        # print(len(local_image_array))
        num_rede = digit_recog(a1, training_instance=i, dataset=dataset)
        # print(f'{num_real} / {num_rede}')
        if num_rede != np.nan:
            if (num_real == num_rede):
                cont_acert += 1

    return 100 * cont_acert / len(dataset)

def main():
    rnd_seed = np.random.seed(10)

    # cria rede neural
    L = 4
    m = [(28 * 28), 512, 256, 128, 10]
    a = [1, 1., 1., 1.]
    b = [2/3, 2/3, 2/3, 2/3]
    a1 = nnc.rede_neural(L, m, a, b)

    print(a1.m[-1])

    # Base de dados de treinamento
    # Se for utilizar o Jupyter notebook, utilizar a linha abaixo
    # dataset = pd.read_csv('mnist_test.csv')
    dataset = pd.read_csv('mnist_test.csv')

    dataset = dataset.iloc[0:100]
    dataset.iloc[:, 1:-1] = dataset.iloc[:, 1:-1] / 255
    dataset.iloc[:, 1:-1] = dataset.iloc[:, 1:-1] * 2. - 1.
    test_dataset = dataset
    n_inst = len(dataset.index)
    dataset.head()

    # parâmetros de treinamento da rede
    n_epoch = 10
    n_inst = len(dataset.index)
    N = n_inst * n_epoch
    step_plot = int(N / 10)

    # taxa de treinamento
    eta = np.ones((L, N))
    eta[0] = list(np.linspace(10., 1., N))
    eta[1] = list(np.linspace(10., 0.001, N))
    eta[2] = list(np.linspace(10., 0.002, N))
    eta[3] = list(np.linspace(10., 0.004, N))
    plt.figure()
    for l in range(0, L):
        plt.plot(eta[l])
        pass
    eta = np.transpose(eta)

    # Inicializa os pesos com valores aleatórios
    for l in range(0, L):
        a1.l[l].w = np.random.rand(m[l + 1], m[l] + 1) * 1.0 - 0.5
        # Inicializa o Bias como zero
        for j in range(0, m[l + 1]):
            a1.l[l].w[j][-1] = 0

    # Termo momentum
    alpha = np.ones((L, N))
    alpha[0] *= 0.00000  # camada de entrada
    alpha[1] *= 0.00000  # camada oculta 1
    alpha[2] *= 0.00000  # camada de saída
    alpha[3] *= 0.00000  # camada de saída

    alpha = np.transpose(alpha)

    # Vetor de pesos para plotar gráficos de evolução deles.
    a1plt = list()

    teste_acertividade(dataset,a1)

    Eav = np.ones(n_epoch)
    Eav[-1] = 1
    Eav[-2] = 1
    # início do treinamento
    print(f'Iniciado treinamento')
    acertividade = np.zeros(n_epoch)
    for ne in range(0, n_epoch):
        dataset_shufle = dataset.sample(frac=1, random_state=rnd_seed, axis=0)
        e_epoch = 0
        for ni in range(0, n_inst):
            n = ni + ne * (n_inst)
            if n >= (N - 1):
                break
            x = list(dataset_shufle.iloc[ni, 1:(m[0] + 1)])

            d = np.ones(m[L]) * -1
            num = dataset_shufle.iloc[ni, 0]
            d[num] = 1

            a1.forward_propagation(x=x)
            a1.backward_propagation(x=x, d=d, alpha=alpha[n], eta=eta[n])

            if n >= step_plot:
                if (n % step_plot == 0):

                    if Eav[ne - 1] < Eav[ne - 2]:
                        sentido = "Down"
                    else:
                        sentido = "up"

                    print(f'Época: {ne}/{n_epoch}, Inst. {ni}/{n_inst}, n={n}/{N} Erro: {Eav[ne - 1]} {sentido}')
                    temp_rede = nnc.rede_neural(L, m, a, b)
                    for l in range(0, L):
                        temp_rede.l[l].w = np.copy(a1.l[l].w)
                    a1plt.append(temp_rede)

            e_epoch += a1.get_sum_eL() ** 2
        Eav[ne] += Eav[ne] + 1 / (2 * N) * e_epoch
        # acertividade[ne] = teste_acertividade()
        if (Eav[ne] < 1E-7):
            print(f'Erro mínimo: {Eav[ne]}')
            break
    Eav[-2] = Eav[ne - 3]
    Eav[-1] = Eav[ne - 3]

    teste_acertividade(dataset,a1)
    plt.figure()
    plt.title('Acertividade')
    plt.plot(acertividade)

    buttonPressed = False
    while not buttonPressed:
        buttonPressed = plt.waitforbuttonpress()

    print(f'Épocas necessárias: {n}')
    plt.figure()
    plt.plot(Eav)
    plt.title('Erro quadrático médio por época')

    buttonPressed = False
    while not buttonPressed:
        buttonPressed = plt.waitforbuttonpress()

    # Extrai os pesos da rede neural relativos ao primeiro neurônio de cada camada
    wl0n0 = np.zeros((len(a1plt), m[0] + 1))
    wl1n0 = np.zeros((len(a1plt), m[1] + 1))
    wl2n0 = np.zeros((len(a1plt), m[2] + 1))
    for i in range(0, len(a1plt)):
        wl0n0[i] = a1plt[i].l[0].w[0]
        wl1n0[i] = a1plt[i].l[1].w[0]
        wl2n0[i] = a1plt[i].l[2].w[0]
    plt.figure()
    lineObjects = plt.plot(wl0n0)
    plt.legend(iter(lineObjects), ('W0', 'W1', 'Bias'))
    plt.title('Pesos do primeiro neurônio da camada de Entrada')

    buttonPressed = False
    while not buttonPressed:
        buttonPressed = plt.waitforbuttonpress()

    plt.figure()
    lineObjects = plt.plot(wl1n0)
    plt.legend(iter(lineObjects), ('W0', 'W1', 'W2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'Bias'))
    plt.title('Pesos do primeiro neurônio da primeira camada oculta')
    buttonPressed = False
    while not buttonPressed:
        buttonPressed = plt.waitforbuttonpress()

    plt.figure()
    plt.plot(wl2n0)
    plt.legend(iter(lineObjects), ('W0', 'W1', 'Bias'))
    plt.title('Pesos do primeiro neurônio da camada de saída')
    buttonPressed = False
    while not buttonPressed:
        buttonPressed = plt.waitforbuttonpress()
    # Teste da rede neural

    acertividade = teste_acertividade(dataset,a1)
    print(f'Acertividade = {acertividade:.3f}%')

    buttonPressed = False
    while not buttonPressed:
        buttonPressed = plt.waitforbuttonpress()

    for i in range(0, 10):
        teste2(dataset, i, a1)
        print("")

    # Imprime os pesos obtidos após o treinamento
    print(f'Função de ativação: Tangente hiperbólica')
    print(f'Quantidade de nós por camada: {m}')
    print(f'Profundidade de rede: {L}')
    print(f'Rede totalmente conectada')
    for l in range(0, L):
        print(f'\n Layer {l}')
        print(a1.l[l].w)
        print(f'Parâmetros da função de ativação: a= {a1.a[l]}, b= {a1.b[l]}')

def teste2(dataset, pos_dataset, rede):
    im_array = list(dataset.iloc[pos_dataset, 1:(rede.m[0] + 1)])
    # digit_recog()
    display_number(dataset, pos_dataset, save=True)

    num_treino = digit_recog(rede, image_array=im_array)
    num_correto = dataset.iloc[pos_dataset, 0]

    print(f'num_treino = {num_treino}, num_correto = {num_correto}')
    # print(ylL[n])




if __name__=='__main__':
  main()