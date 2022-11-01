import numpy as np




def main():
    fitness_list = [5,	2,	3,	1,	6,	4,	9,	100,	3,	4]
    fitness_prob = [0] * len(fitness_list)
    # a= rotate_wheel(fitness_list)
    lim_iter = 100000
    for i in range(0,lim_iter):
        fitness = rotate_wheel(fitness_list)
        position = get_position(fitness_list, fitness)
        fitness_prob[position] += 1/lim_iter

    str_retult = '['
    for i in range(0,len(fitness_list)):
        str_retult = f'{str_retult}{(fitness_prob[i] * 100):.2f}%, '
    str_retult = f'{str_retult[0:-2]}]'
    print(f'{str_retult}')

def get_position(fitness_list, fitness):
    return_value = np.nan
    for i in range(0, len(fitness_list)):
        if fitness_list[i] == fitness:
            return_value = i
            break
    return return_value

def rotate_wheel(fitness_list):
    return_value = 0
    fitness_list_sum = [0] * len(fitness_list)
    fitness_list_sort = np.sort(fitness_list)

    val_ant = 0
    for i in range(0, len(fitness_list)):
        fitness_list_sum[i] = val_ant + fitness_list[i]
        val_ant = fitness_list_sum[i]
    a = np.random.randint(np.min(fitness_list_sum), np.max(fitness_list_sum))

    position = 0

    for i in range(0,len(fitness_list_sum)):
        if a <= fitness_list_sum[i]:
            position = i
            break



    # for i in range(0,len(fitness_list)):
    #     if fitness_list[i] == fitness_list_sort[position]:
    #         return_value = fitness_list[i]

    # print(f'{fitness_list}')
    # print(f'{fitness_list_sort}')
    # print(fitness_list_sum)
    # print(f'sorteio={a}, i={i}, position={position}, fitness sorteado={fitness_list[position]}')

    return fitness_list[position]



if __name__ == '__main__':
    main()