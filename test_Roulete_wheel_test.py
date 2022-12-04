from unittest import TestCase
import numpy as np
import classe_rede_neural as nnc


class Test(TestCase):
    def test_rotate_roulette_wheel(self):
        mode = nnc.selection_parents_mode.ROULETTE_WHEEL
        fitness_list = [1., .02, .03, .01, .06, .04, .09, .0100, .03, .04]
        # fitness_list = [4]  * len(fitness_list)

        # exit()
        if mode == nnc.selection_parents_mode.RANK_SELECTION:
            probs = self.get_rank_array(fitness_list) / np.sum(np.arange(1, len(fitness_list) + 1))
        elif mode == nnc.selection_parents_mode.ROULETTE_WHEEL:
            probs = fitness_list / np.sum(fitness_list)


        fitness_prob = [0] * len(fitness_list)
        # a= rotate_wheel(fitness_list)
        lim_iter = 10000
        for i in range(0, lim_iter):
            position = nnc.rotate_roulette_wheel(fitness_list, mode=mode)

            fitness_prob[position] += 1 / lim_iter

        err = fitness_prob - probs
        sum_probs = np.sum(fitness_prob)
        max_err = np.max(err)
        print(f'First parent:')
        print(f'Max error: {max_err}, sum of probabilities obtained: {sum_probs}')
        print(f'{self.format_array(probs)}- fitness prob calculated')
        print(f'{self.format_array(fitness_prob)} - fitness prob real')

        print(f'{self.format_array(err)} - error')

        print(self.format_array(fitness_list, percent=False))
        parent1_id = nnc.rotate_roulette_wheel(fitness_list)
        fitness_list.pop(parent1_id)
        print(self.format_array(fitness_list, percent=False))

        print(f'\nSecond parent:')
        parent2_id = nnc.rotate_roulette_wheel(fitness_list)
        print(f'Parent1: {parent1_id}, Parent2: {parent2_id}')

        self.assertTrue(expr=(np.abs(max_err) < 0.01),msg="Error outside allowed limit")
        # self.fail()

    def get_rank_array(self,array):
        positions = np.arange(1, len(array)+1)
        table = np.dstack((array, positions))[0]
        table = table[table[:, 0].argsort()]
        table[:,0]=positions
        table = table[table[:, 1].argsort()]

        # print(table)
        # print(array)
        # print(table[:,0])
        return table[:,0]

    def format_array(self,array, percent=True):
        str_percent = ''
        percent_multiplier = 1
        if percent == True:
            str_percent = '%'
            percent_multiplier = 100

        str_result = '['
        for i in range(0, len(array)):
            str_result = f'{str_result}{(array[i] * percent_multiplier):.2f}{str_percent}, '
        str_result = f'{str_result[0:-2]}]'
        return str_result