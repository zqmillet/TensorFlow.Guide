import numpy as np
from random import uniform

A = 1
B = 0

def classify(x1, x2):
    y = x1 - 3 * x2 + 1
    return A if y > 0 else B

def main():
    with open('./training.dat', 'w', encoding = 'utf8') as file:
        file.write('x1, x2, type\n')
        for _ in range(500):
            x1 = uniform(-1, 1)
            x2 = uniform(-1, 1)
            file.write('{x1}, {x2}, {type}\n'.format(x1 = x1, x2 = x2, type = classify(x1, x2)))

if __name__ == '__main__':
    main()