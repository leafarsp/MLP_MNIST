from sklearn.neural_network import MLPClassifier
import pandas as pd
import matplotlib.pyplot as plt
import classe_rede_neural as nnc
import numpy as np


def main():
    a1 = nnc.load_neural_network('MNISTS_BackProp.xlsx')
    clf = a1.convert_model_to_SciKitLearning()

    n_inst = 500
    n_class=10
    # Base de dados de treinamento
    # Se for utilizar o Jupyter notebook, utilizar a linha abaixo
    # dataset = pd.read_csv('mnist_test.csv')
    print(f'Loading dataset')
    dataset = pd.read_csv('mnist_train_small.csv')

    # Filtrando apenas o número 1
    # dataset = dataset.loc[dataset['7'] == 1]
    # dataset = dataset[dataset['6'].isin([1,4])]
    print(f'Adapting dataset')
    dataset = dataset.iloc[0:n_inst]

    dataset.iloc[:, 1:] = dataset.iloc[:, 1:] / 255
    dataset.iloc[:, 1:] = dataset.iloc[:, 1:] * 2. - 1.

    print(f'Loading and adapting test dataset')
    test_dataset = pd.read_csv('mnist_test.csv')
    test_dataset = test_dataset.iloc[0:n_inst]
    test_dataset.iloc[:, 1:] = test_dataset.iloc[:, 1:] / 255
    test_dataset.iloc[:, 1:] = test_dataset.iloc[:, 1:] * 2. - 1.

    X = dataset.iloc[:,1:].to_numpy()


    print(np.shape(X))

    y=[[0]*n_class]*n_inst

    for i in range(0,n_inst):
        y[i] = list(output_layer_activation(output_value=dataset.iloc[i,0],num_classes=n_class))

    print(np.shape(y))


    if clf is not None:
        #clf.hidden_layer_sizes = (15,),

        clf.activation = 'tanh'

        clf.solver = 'sgd'
        clf.alpha = 1e-5
        clf.batch_size = 'auto'
        clf.learning_rate = 'adaptive'
        clf.learning_rate_init = 0.8
        clf.power_t = 0.5
        clf.max_iter = 20000000
        clf.shuffle = True
        clf.random_state = 1
        clf.tol = 0.0001
        clf.verbose = True
        clf.warm_start = False
        clf.momentum = 0.9
        clf.nesterovs_momentum = True
        clf.early_stopping = False
        clf.validation_fraction = 0.1
        clf.beta_1 = 0.9
        clf.beta_2 = 0.999
        clf.epsilon = 1e-08
        clf.n_iter_no_change = 10000000
        clf.max_fun = 15000
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=(15,),
            activation='tanh',
            # *,
            solver='sgd',
            alpha=1e-5,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.8,
            power_t=0.5,
            max_iter=200000,
            shuffle=True,
            random_state=1,
            tol=0.0001,
            verbose=True,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=False,
            validation_fraction=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            n_iter_no_change=10000000,
            max_fun=15000
        )
    clf.fit(X, y)
    count = 0
    for i in range(0, n_inst):
        a = clf.predict([X[i]])[0]
        # print(a)
        predicted_number = get_output_class(y[i])
        obtained_number = get_output_class(a)
        result_str = ''
        if obtained_number == predicted_number:
            count +=1
            result_str = ' Wrong'
        print(f'Predicted_number: {predicted_number}, Real number:{obtained_number}{result_str}')
    acertividade= count / n_inst
    print(f'Acertividade:{acertividade*100}%')
        # print(f'Predicted_number: {get_output_class(a)}')
        # print(f'Real number:{get_output_class(y[i])}')

    print(clf.coefs_)

    print(clf.hidden_layer_sizes)

    print(clf.get_params())

    for i in range(0, len(clf.coefs_)):
        print(f'\n Layer {i}')
        print(np.transpose(clf.coefs_[i]))
        print(f'Bias:{clf.intercepts_[i]}')

    a1 = nnc.load_scikit_model(clf)
    a1.save_neural_network('teste_MNIST_scikit_learn.xlsx')

    # nnc.teste_neural_network(dataset, a1)
    # acertividade = nnc.teste_acertividade(dataset, n_class, a1, True)
    # print(f'Acertividade:{acertividade}')
def output_layer_activation(output_value, num_classes):
    d = np.ones(num_classes, dtype=np.float64) * -1.
    # num = dataset_shufle.iloc[ni, 0]
    d[output_value] = 1.
    return d


def get_output_class(output_neurons:list, threshold=0.8):
    num_out = np.nan
    cont_neuronio_ativo = 0
    for j in range(0, len(output_neurons)):
    # for j in range(self.m[self.L]-1, -1, -1):
        if (output_neurons[j] > (1 * threshold)):
            # num_out = j
            num_out = j
            cont_neuronio_ativo += 1
        if (cont_neuronio_ativo > 1):
            num_out = np.nan
            break
    return num_out















if __name__ == '__main__':
    main()


