from unittest import TestCase
import test_Roulete_wheel_test as test
from enum import Enum


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

def main():
    test(a=Color.BLUE)
    # test_case = TestCase(methodName='teste')
    # test.test_rotate_roulette_wheel(test_case)



def test (a:Color):
    print(a)


if __name__ == '__main__':
    main()