#!/usr/bin/env python3
"""create simple line plot, line col red
    """

import numpy as np
import matplotlib.pyplot as plt


def line():
    """ plot y as a line graph

    Returns:
        _type_: _description_
    """

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, color='r')  # set color
    plt.xlim(0, 10)  # set x limit
    return plt.show()  # return printed
