import itertools

import numpy as np


def train_perceptron(eta, target, input, output, weights, theta):
    weights = weights + eta * (target - output) * np.array(input)
    theta = theta - eta * (target - output)
    return weights, theta

    # sgn function, if 0 value is set to 1


def sgn(value):
    return 1 if value >= 0 else -1


def mcculloch_pitts(weights, input, theta):
    bm = np.dot(weights, input) - theta  # bm is the neurons local field
    output = sgn(bm)
    return output


def generate_boolean_functions(n):
    inputs = list(itertools.product([0, 1], repeat=n))

    if n in (2, 3):
        targets = list(itertools.product([-1, 1], repeat=2**n))
        return inputs, targets
    else:
        return inputs


def linearly_separable(inputs, target, weights, theta):
    for pattern, t in zip(inputs, target):
        output = mcculloch_pitts(weights, pattern, theta)
        if output != t:
            return False
    return True


dimensions = [2, 3, 4, 5]
epochs = 20
eta = 0.05
sample_size = 10000
theta = 0
trials = 20

for n in dimensions:
    no_of_functions = 2 ** (2**n)

    # known number of linearly separable functions
    if n == 2:
        actual_no_of_separable = 14
    elif n == 3:
        actual_no_of_separable = 104
    elif n == 4:
        actual_no_of_separable = 1882
    elif n == 5:
        actual_no_of_separable = 94572

    fraction_list = []

    if n in (2, 3):
        inputs, targets = generate_boolean_functions(n)
        for trial in range(trials):
            separable = 0
            for target in targets:
                weights = np.random.normal(loc=0.0, scale=np.sqrt(1 / n), size=n)
                theta = 0
                for iterations in range(epochs):
                    for pattern, t in zip(
                        inputs, target
                    ):  # train once for every input output pair
                        output = mcculloch_pitts(weights, pattern, theta)
                        weights, theta = train_perceptron(
                            eta, t, pattern, output, weights, theta
                        )
                    if linearly_separable(inputs, target, weights, theta):
                        separable += 1
                        break
            fraction_separable = separable / (no_of_functions)
            fraction_list.append(fraction_separable)

    elif n in (4, 5):
        inputs = generate_boolean_functions(n)
        for trial in range(trials):
            separable = 0
            for sample_function in range(sample_size):
                target = np.random.choice(
                    [-1, 1], size=2**n
                )  # generate random target vector
                weights = np.random.normal(loc=0.0, scale=np.sqrt(1 / n), size=n)
                theta = 0
                for iterations in range(epochs):
                    for pattern, t in zip(
                        inputs, target
                    ):  # train once for every input output pair
                        output = mcculloch_pitts(weights, pattern, theta)
                        weights, theta = train_perceptron(
                            eta, t, pattern, output, weights, theta
                        )
                    if linearly_separable(inputs, target, weights, theta):
                        separable += 1
                        break
            fraction_separable = separable / (sample_size)
            fraction_list.append(fraction_separable)

    average_fraction = np.mean(fraction_list)
    std_fraction = np.std(fraction_list, ddof=1)
    actual_fraction = actual_no_of_separable / no_of_functions

    print(
        f"--- Results for n = {n} ---\n"
        f"Computed Fraction : {average_fraction:.6f}\n"
        f"Standard Deviation: {std_fraction:.6f}\n"
        f"Actual Fraction   : {actual_fraction:.6f}"
    )
