from unittest import TestCase
import numpy as np
import classe_rede_neural as nnc


class Testrede_neural(TestCase):
    def test_output_layer_activation(self):
        L = 2
        m = [2, 2, 10]
        a = [0.9, 0.9]
        b = [0.5, 0.5]
        a1 = nnc.rede_neural(L, m, a, b)
        num_classes = 10
        for i in range(0,num_classes):
            num_real = i
            d = a1.output_layer_activation(num_real, num_classes)
            print(d)

        # self.fail()
