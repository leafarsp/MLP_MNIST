import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import classe_rede_neural as nnc
import cv2 as cv2


def main():

    n_inst = 1000
    num_classes = 10
    rnd_seed = 0

    population = list()
    # rede 1
    L = 2
    m = [(28 * 28), 15, 10]
    a = [1.7, 0.9]
    b = [0.002, 1.]
    population.append(nnc.rede_neural(L, m, a, b))
    population[-1].initialize_weights_random(2.)
    population[-1].set_id(len(population)-1)

    # rede 2
    L = 3
    m = [(28 * 28), 30, 15, 10]
    a = [1.7, 0.9,0.9]
    b = [0.002, 1.,1.]
    population.append(nnc.rede_neural(L, m, a, b))
    population[-1].initialize_weights_random(2.)
    population[-1].set_id(len(population) - 1)

    # rede 3
    L = 3
    m = [(28 * 28), 20, 10, 10]
    a = [1.7, 0.9, 0.9]
    b = [0.002, 1., 1.]
    population.append(nnc.rede_neural(L, m, a, b))
    population[-1].initialize_weights_random(2.)
    population[-1].set_id(len(population) - 1)

    # rede 4
    L = 2
    m = [(28 * 28), 25, 10]
    a = [1.7, 0.9]
    b = [0.002, 1.]
    population.append(nnc.rede_neural(L, m, a, b))
    population[-1].initialize_weights_random(2.)
    population[-1].set_id(len(population) - 1)

    # carregando dataset
    print(f'Loading dataset')
    dataset = pd.read_csv('mnist_train_small.csv')

    # Filtrando apenas o número 1
    # dataset = dataset.loc[dataset['7'] == 1]
    # dataset = dataset[dataset['6'].isin([1,4])]
    print(f'Adapting dataset')
    dataset = dataset.iloc[0:n_inst]

    dataset.iloc[:, 1:-1] = dataset.iloc[:, 1:-1] / 255
    dataset.iloc[:, 1:-1] = dataset.iloc[:, 1:-1] * 2. - 1.

    print(f'Loading and adapting test dataset')
    test_dataset = pd.read_csv('mnist_test.csv')
    test_dataset = test_dataset.iloc[0:n_inst]
    test_dataset.iloc[:, 1:-1] = test_dataset.iloc[:, 1:-1] / 255
    test_dataset.iloc[:, 1:-1] = test_dataset.iloc[:, 1:-1] * 2. - 1.



    print(f'Population playing')
    nnc.population_play_concurent(
        dataset=dataset,
        test_dataset=test_dataset,
        num_classes=num_classes,
        population=population,
        rede=population[0],
        rnd_seed=rnd_seed,
        generation=0,
        best_ind=population[0],
        acertividade=0)

    print(f'Fitness list:')
    fitness_list = nnc.get_fitness_list(population)
    best_ind = nnc.get_best_ind(population, 0)
    print(fitness_list)
    print(f'best fitness: {best_ind.get_fitness()}')



if __name__ == '__main__':
    main()