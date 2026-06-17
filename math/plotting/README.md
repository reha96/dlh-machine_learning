# Math — Plotting

A progressive study of data visualization with Matplotlib — from basic line plots to 3D PCA, covering line graphs, scatter plots, histograms, bar charts, subplots, color mapping, and dimensionality reduction.

---

## Learning Objectives

| # | Concept |
|---|---------|
| 1 | Create a basic line plot with `plt.plot()` and customize colors |
| 2 | Generate scatter plots with axis labels, titles, and point colors |
| 3 | Apply logarithmic scaling to axes with `plt.yscale("log")` |
| 4 | Plot multiple curves on the same figure with legends and line styles |
| 5 | Build histograms with custom bin edges and bar outlines |
| 6 | Arrange multiple subplots in a grid with `subplot()` |
| 7 | Create stacked bar charts with custom colors and legends |
| 8 | Add colorbars to scatter plots for a third data dimension |
| 9 | Visualize high-dimensional data in 3D after PCA reduction |

---

## Task-by-Task Reference

Each task below highlights the **unique challenge** it posed and the **new technique** introduced — techniques from earlier tasks are not repeated.

---

### Task 0 — Line Plot (`0-line.py`)

**Challenge:** Create a basic line graph — the simplest possible visualization, introducing the Matplotlib plotting pipeline.

**Approach:** Generate y-values as the cube of 0–10 using NumPy, then call `plt.plot(y, color='r')` to draw a red line. Set x-axis limits with `plt.xlim()`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `plt.plot(y, color='r')` | Create a line plot with a specific color |
| `plt.xlim(0, 10)` | Set the visible range of the x-axis |
| `plt.figure(figsize=(w, h))` | Set the figure dimensions in inches |
| `plt.show()` | Render and display the plot |

> **Key takeaway:** `plt.plot()` is the universal line-plotting function. The `color` parameter accepts named colors (`'r'`), hex codes, or RGB tuples. `plt.show()` renders everything — without it, no plot appears.

---

### Task 1 — Scatter Plot (`1-scatter.py`)

**Challenge:** Create a labeled scatter plot from bivariate normal data — introducing axis labels, titles, and scatter-specific styling.

**Approach:** Generate 2000 points from a multivariate normal distribution with `np.random.multivariate_normal()`. Use `plt.scatter(x, y, c='magenta')` for colored points. Add labels and title with `plt.xlabel()`, `plt.ylabel()`, `plt.title()`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.random.multivariate_normal(mean, cov, n)` | Generate correlated bivariate data |
| `plt.scatter(x, y, c='magenta')` | Create a scatter plot with colored markers |
| `plt.xlabel("label")`, `plt.ylabel("label")` | Label the x and y axes |
| `plt.title("title")` | Add a title above the plot |

> **Key takeaway:** `plt.scatter()` is ideal for showing relationships between two variables. Unlike `plt.plot()`, it doesn't connect points — each marker stands alone. Always label your axes for interpretable visualizations.

---

### Task 2 — Logarithmic Scale (`2-change_scale.py`)

**Challenge:** Visualize exponential decay data where values span multiple orders of magnitude — introducing logarithmic axis scaling.

**Approach:** Model Carbon-14 decay as $y = e^{(\ln 0.5 / 5730) \cdot x}$ for $x$ from 0 to 28,650 years. Apply `plt.yscale("log")` to compress the y-axis, making the exponential curve appear linear — a classic diagnostic for exponential processes.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `plt.yscale("log")` | Transform the y-axis to logarithmic scale |
| `np.exp()` for exponential modeling | Compute $e^{kx}$ for radioactive decay |
| `np.log(0.5)` for half-life constant | Derive the decay rate from half-life |

> **Key takeaway:** Logarithmic scales make exponential relationships visually linear — a straight line on a log plot confirms exponential behavior. Use `plt.yscale("log")` or `plt.xscale("log")` when data spans multiple orders of magnitude.

---

### Task 3 — Two Curves with Legend (`3-two.py`)

**Challenge:** Compare two radioactive decay curves (C-14 and Ra-226) on the same plot — introducing multiple line styles, legends, and axis range control.

**Approach:** Plot C-14 with a red dashed line (`linestyle="--"`) and Ra-226 with a green solid line. Use `label=` on each `plt.plot()` call, then `plt.legend(loc="upper right")` to identify the curves. Set both x and y limits with `plt.xlim()` and `plt.ylim()`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `linestyle="--"` and `linestyle="-"` | Control line appearance (dashed, solid, dotted) |
| `label="C-14"` in `plt.plot()` | Name a curve for the legend |
| `plt.legend(loc="upper right")` | Display a legend at a specified position |
| `plt.ylim(0, 1)` | Set the visible range of the y-axis |

> **Key takeaway:** When plotting multiple curves, always add a legend. The `label` parameter in `plt.plot()` pairs with `plt.legend()` to identify each curve. Line styles (solid, dashed, dotted) provide redundancy for colorblind accessibility.

---

### Task 4 — Histogram (`4-frequency.py`)

**Challenge:** Visualize the distribution of student grades — introducing histograms with custom bin edges and styling.

**Approach:** Generate 50 random grades from $\mathcal{N}(68, 15)$. Define bins every 10 units (0–100) with `np.arange()`. Call `plt.hist()` with `bins=bins` and `edgecolor='black'` for outlined bars. Set x-ticks to match bin edges.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `plt.hist(data, bins=bins, edgecolor='black')` | Create a histogram with specified bins and bar outlines |
| `np.arange(0, 101, 10)` for bin edges | Define exact bin boundaries |
| `plt.xticks(bins)` | Position x-axis tick marks at bin edges |
| `np.random.normal(mean, std, n)` | Generate normally distributed random data |

> **Key takeaway:** Histograms reveal the shape of data distributions (normal, skewed, bimodal). The `bins` parameter controls granularity — too few bins hide detail, too many create noise. `edgecolor` makes individual bars distinguishable.

---

### Task 5 — Subplot Grid (`5-all_in_one.py`)

**Challenge:** Combine all five previous plots into a single figure — introducing Matplotlib's subplot grid system and figure-level titling.

**Approach:** Create a 3×2 grid of subplots using `plt.subplot(3, 2, i)`. The last plot spans two columns using `plt.subplot2grid()`. Set all font sizes to `x-small` with `plt.rcParams`. Add a figure-level title with `plt.suptitle()` and adjust spacing with `plt.tight_layout()`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `plt.subplot(rows, cols, index)` | Arrange multiple plots in a grid |
| `plt.subplot2grid()` for spanning cells | Make a plot occupy multiple grid cells |
| `plt.suptitle("title")` | Add a title to the entire figure (not individual subplot) |
| `plt.tight_layout(pad=3, w_pad=0.5, h_pad=1.0)` | Automatically adjust subplot spacing |
| `plt.rcParams` for font sizing | Set global style parameters |

> **Key takeaway:** Subplots let you compare multiple visualizations side by side. `plt.subplot()` uses 1-based indexing. For complex layouts, `subplot2grid()` gives fine-grained control over cell spanning.

---

### Task 6 — Stacked Bar Chart (`6-bars.py`)

**Challenge:** Display categorical data as stacked bars — introducing bar charts, custom colors, and categorical tick labels.

**Approach:** Use `plt.bar()` with `bottom=` parameter to stack fruit quantities per person. Assign specific colors (apples=red, bananas=yellow, oranges=#ff8000, peaches=#ffe5b4). Add a legend, set bar width to 0.5, and use `plt.xticks()` with person names.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `plt.bar(x, height, bottom=prev, width=0.5, color=c, label=l)` | Create stacked bar segments |
| `bottom` parameter for stacking | Place a bar on top of the previous cumulative height |
| Hex color codes (`#ff8000`, `#ffe5b4`) | Specify custom colors beyond named colors |
| `plt.xticks(positions, labels)` | Replace numeric tick positions with text labels |

> **Key takeaway:** Stacked bar charts show both individual contributions and totals. The `bottom` parameter stacks bars by specifying where each segment starts. Custom hex colors and legends make categorical data readable.

---

### Task 7 — Gradient Scatter (`100-gradient.py`)

**Challenge:** Add a third dimension (elevation) to a scatter plot using color — introducing colorbars for continuous data mapping.

**Approach:** Generate random x, y coordinates and compute elevation $z = 40 - \sqrt{x^2 + y^2} + \text{noise}$. Pass `c=z` to `plt.scatter()` to color points by elevation. Add `plt.colorbar(label="elevation (m)")` for a reference scale.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `plt.scatter(x, y, c=z)` | Map a third variable to point colors |
| `plt.colorbar(label="label")` | Display a color scale bar with a label |
| Automatic color mapping | Matplotlib maps the `c` array to a default colormap (viridis) |

> **Key takeaway:** Color adds a third dimension to 2D scatter plots. The `c` parameter accepts any numeric array — Matplotlib automatically normalizes and maps values to a colormap. Always include a colorbar to make the scale interpretable.

---

### Task 8 — 3D PCA Visualization (`101-pca.py`)

**Challenge:** Visualize 4-dimensional Iris flower data in 3D — introducing PCA dimensionality reduction and 3D plotting.

**Approach:** Load the Iris dataset, center the data by subtracting means, compute SVD to get principal components, then project onto the top 3 components. Use `Axes3D` for 3D scatter with `cmap='plasma'` coloring by species label.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `np.linalg.svd()` for PCA | Singular Value Decomposition extracts principal components |
| Data centering: `data - data_means` | Mean-center before PCA for correct component directions |
| `np.matmul(norm_data, Vh[:3].T)` | Project data onto top 3 principal components |
| `ax = fig.add_subplot(projection='3d')` | Create a 3D axes for volumetric visualization |
| `cmap='plasma'` | Use a specific colormap for categorical coloring |

> **Key takeaway:** PCA reduces high-dimensional data to its most informative dimensions. SVD is the computational engine — it finds the directions of maximum variance. 3D plots let you inspect the reduced space interactively. PCA is essential for both visualization and preprocessing.

---

## Technique Inventory

| Task | New technique summarized | Category |
|------|--------------------------|----------|
| 0 | `plt.plot()`, `plt.xlim()`, `plt.show()` | Basic Plotting |
| 1 | `plt.scatter()`, axis labels, title, multivariate normal data | Scatter Plots |
| 2 | `plt.yscale("log")`, exponential decay modeling | Scale & Transform |
| 3 | Multiple curves, `linestyle`, `plt.legend()`, `plt.ylim()` | Multi-Curve |
| 4 | `plt.hist()`, custom bins, `edgecolor`, `plt.xticks()` | Distributions |
| 5 | `plt.subplot()`, `subplot2grid()`, `suptitle()`, `tight_layout()` | Subplots |
| 6 | `plt.bar()` with `bottom`, hex colors, categorical ticks | Bar Charts |
| 7 | `plt.scatter(c=z)`, `plt.colorbar()`, 3D color mapping | Color Mapping |
| 8 | PCA via `np.linalg.svd()`, 3D projection, `Axes3D`, `cmap` | PCA & 3D |

---

## Resources

- [Matplotlib Pyplot Tutorial](https://matplotlib.org/stable/tutorials/pyplot.html)
- [Matplotlib Colors API](https://matplotlib.org/stable/gallery/color/named_colors.html)
- [NumPy SVD Documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
- [PCA Explained — towardsdatascience](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)
- [Iris Dataset — Wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set)