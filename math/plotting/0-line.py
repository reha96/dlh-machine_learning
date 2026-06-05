#!/usr/bin/env python3
"""create simple line plot, line col red
    """


def line():
    """ plot y as a line graph

    Returns:
        _type_: _description_
    """

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, color='r')
    return plt.show()
