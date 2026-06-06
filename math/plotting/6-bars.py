#!/bin/usr/env python3
"""Complete the following source code to plot a stacked bar graph:

fruit is a matrix representing the number of fruit various
people possess
The columns of fruit represent the number of fruit Farrah,
Fred, and Felicia have, respectively
The rows of fruit represent the number of apples, bananas,
oranges, and peaches, respectively
The bars should represent the number of fruit each person
possesses:
The bars should be grouped by person, i.e, the horizontal
axis should have one labeled tick per person
Each fruit should be represented by a specific color:
apples = red
bananas = yellow
oranges = orange (#ff8000)
peaches = peach (#ffe5b4)
A legend should be used to indicate which fruit is
represented by each color
The bars should be stacked in the same order as the
rows of fruit, from bottom to top
The bars should have a width of 0.5
The y-axis should be labeled Quantity of Fruit
The y-axis should range from 0 to 80 with ticks every
10 units
The title should be Number of Fruit per Person

"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    names = ('Farrah', 'Fred', 'Felicia')
    fruit_names = ["apples", "bananas", "oranges", "peaches"]
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(3)
    plt.rc('axes', labelsize='medium', titlesize='medium')

    for i in range(len(fruit)):
        ax.bar(names, fruit[i], width, label=fruit_names[i],
               color=colors[i], bottom=bottom)
        bottom += fruit[i]

    ax.set_ylabel("Quantity of Fruit")
    ax.set_title("Number of Fruit per Person")
    ax.set_ylim(0, 80)
    ax.set_yticks(np.arange(0, 81, 10))
    ax.legend(loc="upper right")

    return plt.show()


bars()
