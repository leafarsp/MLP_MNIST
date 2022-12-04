from sklearn.neural_network import MLPClassifier
import pandas as pd
import matplotlib.pyplot as plt
import classe_rede_neural as nnc
import numpy as np






def main():

    high_lim = 1.
    low_lim = -1.
    data = {'y1': [high_lim, low_lim,  low_lim,  high_lim],
            'y2': [low_lim,  high_lim, high_lim, low_lim],
            'x1': [-1., -1., 1., 1.],
            'x2': [-1., 1., -1., 1.]}

    dataset = pd.DataFrame(data=data)
    n_inst = len(dataset.index)

    X = dataset.loc[:, ['x1', 'x2']].to_numpy()
    y = dataset.loc[:, ['y1','y2']].to_numpy()

    dataset.drop(columns=['y1'],inplace=True)
    print(dataset)


#   X = [[0., 0.], [1., 1.]]
#    y = [0, 1]
#     clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                         hidden_layer_sizes=(3,2), random_state=1)

    clf = MLPClassifier(
        hidden_layer_sizes=(2),
        activation='tanh',
        # *,
        solver='sgd', #{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
        alpha=1e-3,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.00001,
        power_t=0.5,
        max_iter=20000000,
        shuffle=True,
        random_state=1,
        tol=1E-9,
        verbose=True,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        n_iter_no_change=10,
        max_fun=15000
    )
    clf.fit(X, y)

    for i in range(0,4):
        a=clf.predict([X[i]])
        print(f'{X[i][0]} XOR {X[i][1]} = {a}')

    print(clf.coefs_)

    print(clf.hidden_layer_sizes)

    print(clf.get_params())

    for i in range(0,len(clf.coefs_)):
        print(f'\n Layer {i}')
        print(np.transpose(clf.coefs_[i]))
        print(f'Bias:{clf.intercepts_[i]}')



    a1 = nnc.load_scikit_model(clf)
    a1.save_neural_network('teste_scikit_learn.xlsx')

    nnc.teste_neural_network(dataset, a1)

    plt_retas(a1, dataset, n_inst)
    # acertividade = nnc.teste_acertividade(dataset, 3, a1)

def plt_retas(rede,dataset, num_inst):
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


if __name__ == '__main__':
    main()