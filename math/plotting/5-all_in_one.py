#!/usr/bin/env python3
"""
Complete the following source code to plot all 5 previous
graphs in one figure:

All axis labels and plot titles should have
a font size of x-small (to fit nicely in one figure)
The plots should make a 3 x 2 grid
The last plot should take up two column widths
(see below)
The title of the figure should be All in One
"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    # ----- Data for plot 0: cubic line -----
    x0 = np.arange(0, 11)
    y0 = x0 ** 3

    # ----- Data for plot 1: scatter (height vs weight) -----
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    # ----- Data for plot 2: exponential decay (C-14, log scale) -----
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    # ----- Data for plot 3: two exponential decays (C-14 and Ra-226) -----
    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    # ----- Data for plot 4: histogram of student grades -----
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    bins = np.arange(0, 101, 10)          # bins every 10 units

    # ----- Global font size settings (x-small for labels and titles) -----
    plt.rc('axes', labelsize='x-small', titlesize='x-small')

    # ----- Create figure with GridSpec (3 rows, 2 columns) -----
    fig = plt.figure()
    gs = fig.add_gridspec(3, 2)

    ax0 = fig.add_subplot(gs[0, 0])   # row0, col0: cubic line
    ax1 = fig.add_subplot(gs[0, 1])   # row0, col1: scatter
    ax2 = fig.add_subplot(gs[1, 0])   # row1, col0: C-14 log scale
    ax3 = fig.add_subplot(gs[1, 1])   # row1, col1: two lines
    ax4 = fig.add_subplot(gs[2, :])   # row2, both columns: histogram

    # ---------- Plot 0: y = x^3 ----------
    ax0.plot(x0, y0, 'r-')
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")

    # ---------- Plot 1: Men's height vs weight (scatter) ----------
    ax1.scatter(x1, y1, c='magenta', marker='o')
    ax1.set_xlabel("Height (in)")
    ax1.set_ylabel("Weight (lbs)")
    ax1.set_title("Men's Height vs Weight")

    # ---------- Plot 2: Exponential decay C-14 (log scale) ----------
    ax2.plot(x2, y2, 'r-')
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Fraction Remaining")
    ax2.set_yscale('log')
    ax2.set_xlim(0, 28650)
    ax2.set_title("Exponential Decay of C-14")

    # ---------- Plot 3: Two radioactive elements ----------
    ax3.plot(x3, y31, color='r', linestyle='--', label='C-14')
    ax3.plot(x3, y32, color='g', label='Ra-226')
    ax3.set_xlabel("Time (years)")
    ax3.set_ylabel("Fraction Remaining")
    ax3.set_xlim(0, 20000)
    ax3.set_ylim(0, 1)
    ax3.set_title("Exponential Decay of Radioactive Elements")
    ax3.legend(loc='upper right')

    # ---------- Plot 4: Histogram of student grades ----------
    ax4.hist(student_grades, bins=bins, edgecolor='black')
    ax4.set_xlabel("Grades")
    ax4.set_ylabel("Number of Students")
    ax4.set_title("Project A")
    ax4.set_xticks(bins)
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 30)

    # Figure title
    fig.suptitle("All in One")

    return plt.show()


if __name__ == "__main__":
    all_in_one()
