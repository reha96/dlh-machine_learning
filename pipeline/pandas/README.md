# Pipeline — Pandas

Data manipulation with pandas — from DataFrame creation through transformation, cleaning, indexing, concatenation, descriptive statistics, and time-series visualization, using real Bitcoin trade data.

---

## Learning Objectives

| # | Concept |
|---|---------|
| 1 | Create DataFrames from NumPy arrays, dictionaries, and CSV files |
| 2 | Rename columns and convert timestamp values to datetime objects |
| 3 | Select columns and slice rows with step indexing |
| 4 | Sort DataFrames by column values in ascending/descending order |
| 5 | Transpose DataFrames (rows ↔ columns) |
| 6 | Remove rows with missing values and fill missing values strategically |
| 7 | Set a column as the DataFrame index |
| 8 | Concatenate DataFrames with hierarchical keys |
| 9 | Filter rows by index range on DataFrames with a MultiIndex |
| 10 | Compute descriptive statistics on numeric columns |
| 11 | Resample time-series data to a lower frequency with aggregation |
| 12 | Visualize resampled data with matplotlib |

---

## Task-by-Task Reference

---

### Task 0 — From NumPy (`0-from_numpy.py`)

**Challenge:** Convert a raw NumPy ndarray into a labeled pandas DataFrame when you do not have column names.

**Approach:** Build the DataFrame from the array, then generate column labels dynamically using ASCII character codes (`chr(65 + i)` for A, B, C, …).

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `pd.DataFrame(array)` | Create a DataFrame from a NumPy ndarray |
| `df.columns = [chr(65 + i) …]` | Assign alphabetical column labels dynamically |
| `chr(65 + i)` | Convert integer index to uppercase letter A–Z |

> **Key takeaway:** DataFrames can wrap raw ndarrays; column labels are always assigned manually when the source has none.

---

### Task 1 — From Dictionary (`1-from_dictionary.py`)

**Challenge:** Create a small DataFrame from scratch with both column headers and named row indices.

**Approach:** Pass a dictionary mapping column names to value lists, plus an explicit `index` parameter.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `pd.DataFrame({…}, index=[…])` | Create a DataFrame from a dict with custom row labels |
| Dict keys → columns | Column order follows dict insertion order |

> **Key takeaway:** Dictionaries are the most readable way to create small inline DataFrames.

---

### Task 2 — From File (`2-from_file.py`)

**Challenge:** Load a real-world CSV dataset when the delimiter is not a comma.

**Approach:** Use `pd.read_csv()` with the `delimiter` parameter to handle any column separator.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `pd.read_csv(filename, delimiter=…)` | Load tabular data from a file with a custom separator |

> **Key takeaway:** `read_csv` is the primary entry point for loading external datasets into pandas.

---

### Task 3 — Rename (`3-rename.py`)

**Challenge:** Clean a raw dataset by renaming columns and converting Unix timestamps to human-readable datetimes.

**Approach:** Use `rename()` to relabel the column, `pd.to_datetime()` with `unit='s'` to convert epoch seconds, then select a subset of columns.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `df.rename(columns={old: new})` | Rename specific columns |
| `pd.to_datetime(series, unit='s')` | Convert Unix epoch seconds to datetime objects |
| `df[["col1", "col2"]]` | Select a subset of columns as a new DataFrame |

> **Key takeaway:** Timestamp columns commonly need both renaming and type conversion before analysis.

---

### Task 4 — Array (`4-array.py`)

**Challenge:** Extract the last several rows of a DataFrame and convert them to a raw NumPy array for use with non-pandas code.

**Approach:** Use `tail(n)` to select the last n rows from specific columns, then `to_numpy()` to get an ndarray.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `df.tail(n)` | Select the last n rows |
| `df.to_numpy()` | Convert a DataFrame (or column subset) to a NumPy ndarray |

> **Key takeaway:** `to_numpy()` is the bridge between pandas and any library that expects raw arrays.

---

### Task 5 — Slice (`5-slice.py`)

**Challenge:** Downsample a high-frequency dataset by selecting every Nth row from specific columns.

**Approach:** Filter columns by name, then apply Python's step-slicing syntax `[::n]` to the DataFrame.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `df[::n]` | Select every n-th row using step slicing |
| `df[["col_a", "col_b"]]` | Column filtering before slicing |

> **Key takeaway:** Python's slice notation works directly on DataFrames for row selection.

---

### Task 6 — Flip Switch (`6-flip_switch.py`)

**Challenge:** Reverse the chronological order of a time-series and then swap rows and columns.

**Approach:** Sort by the timestamp column descending, then use `.T` to transpose.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `df.sort_values("col", ascending=False)` | Sort rows by a column in descending order |
| `df.T` | Transpose rows ↔ columns |

> **Key takeaway:** Sorting and transposing are fundamental reshuffling operations for exploratory data work.

---

### Task 7 — High (`7-high.py`)

**Challenge:** Find the highest values in a dataset by sorting a specific column descending.

**Approach:** Use `sort_values()` with the target column and `ascending=False`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `df.sort_values("col", ascending=False)` | Sort by a single column (descending is explicit) |

> **Key takeaway:** `sort_values` returns a new DataFrame; the original is unchanged unless assigned back.

---

### Task 8 — Prune (`8-prune.py`)

**Challenge:** Remove rows that have missing data in a critical column.

**Approach:** Use boolean indexing with `notna()` to keep only rows where the column has valid values.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `df["col"].notna()` | Boolean mask: True for non-null entries |
| `df[mask]` | Filter rows with a boolean mask |

> **Key takeaway:** Boolean indexing is the standard pattern for row filtering in pandas.

---

### Task 9 — Fill (`9-fill.py`)

**Challenge:** Handle missing values in multiple columns, each with a different fill strategy (drop column, forward-fill, cross-column fill, zero-fill).

**Approach:** Remove `Weighted_Price` with `drop()`, forward-fill `Close` with `ffill()`, backfill `High`/`Low`/`Open` from the `Close` column using `fillna()`, and set volume columns to 0.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `df.drop("col", axis=1)` | Remove a column by name |
| `df["col"].ffill()` | Forward-fill: propagate last valid observation |
| `df["col"].fillna(value)` | Replace NaN with a scalar or Series |
| `df[["a", "b"]].fillna(0)` | Fill multiple columns with the same value |

> **Key takeaway:** Different columns demand different imputation strategies; pandas provides targeted methods for each.

---

### Task 10 — Index (`10-index.py`)

**Challenge:** Promote a data column to become the row index for time-series operations.

**Approach:** Use `set_index()` to designate the `Timestamp` column as the index.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `df.set_index("col")` | Replace the default integer index with a named column |

> **Key takeaway:** A meaningful index (especially datetime) unlocks time-series features like resampling and slicing by label.

---

### Task 11 — Concat (`11-concat.py`)

**Challenge:** Combine two DataFrames from different sources (Coinbase and Bitstamp) into one, labeling each source.

**Approach:** Set the timestamp index on both, filter df2 to a cutoff timestamp, then `pd.concat()` with a `keys` parameter to add source labels.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `pd.concat([df2, df1], keys=["a", "b"])` | Row-bind DataFrames with source labels in a MultiIndex |
| `df[df.index <= cutoff]` | Filter rows by index value |

> **Key takeaway:** `concat` with keys creates a MultiIndex on the first level, preserving provenance.

---

### Task 12 — Hierarchy (`12-hierarchy.py`)

**Challenge:** Merge overlapping time ranges from two sources, swap index levels so timestamp is outermost, and sort chronologically.

**Approach:** Set timestamp index on both, filter to a shared time window, concat with keys, swap the MultiIndex levels, then sort by the timestamp level.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `df.index.swaplevel(0, 1)` | Swap the order of MultiIndex levels |
| `df.sort_index(level=0)` | Sort by a specific MultiIndex level |
| `(df.index >= low) & (df.index <= high)` | Boolean mask for index range filtering |

> **Key takeaway:** MultiIndex manipulation — swaplevel, sort_index — gives fine-grained control over hierarchical data.

---

### Task 13 — Analyze (`13-analyze.py`)

**Challenge:** Get a statistical summary of all numeric columns in one call.

**Approach:** Drop the non-numeric `Timestamp` column, then call `describe()` on the result.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `df.describe()` | Compute count, mean, std, min, quartiles, max for numeric columns |

> **Key takeaway:** `describe()` is the fastest way to assess central tendency and spread of every numeric column.

---

### Task 14 — Visualize (`14-visualize.py`)

**Challenge:** End-to-end data pipeline: load, clean, resample to daily frequency with different aggregations per column, and plot.

**Approach:** Chain drop → rename → datetime conversion → set_index → ffill → fillna → resample with agg dict → matplotlib plot.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `df.resample("D").agg({…})` | Downsample to daily frequency with per-column aggregations |
| `df.plot(subplots=True, figsize=(w, h))` | Quick time-series plot with matplotlib |

> **Key takeaway:** The full pandas workflow — load, clean, reshape, resample, visualize — can be expressed concisely in a single script, enabling rapid data exploration.

---

## Technique Inventory

| Task | New technique summarized | Category |
|------|--------------------------|----------|
| 0 | `pd.DataFrame(array)`, `chr(65 + i)` column labels | DataFrame Creation |
| 1 | `pd.DataFrame({…}, index=…)` from dict | DataFrame Creation |
| 2 | `pd.read_csv(filename, delimiter=…)` | I/O |
| 3 | `rename()`, `pd.to_datetime(…, unit='s')`, column subset | Transformation |
| 4 | `tail(n)`, `to_numpy()` | Transformation |
| 5 | `df[::n]` step slicing | Indexing & Selection |
| 6 | `sort_values(…, ascending=False)`, `.T` transpose | Transformation |
| 7 | `sort_values("col", ascending=False)` | Transformation |
| 8 | `notna()`, boolean mask filtering | Cleaning |
| 9 | `drop()`, `ffill()`, `fillna()` | Cleaning |
| 10 | `set_index("col")` | Indexing & Selection |
| 11 | `pd.concat([…], keys=…)`, index-based filtering | Combining DataFrames |
| 12 | `swaplevel()`, `sort_index(level=…)`, range boolean mask | MultiIndex |
| 13 | `describe()` | Analysis |
| 14 | `resample("D").agg({…})`, `.plot()` | Time Series & Viz |

## Resources

- [pandas Documentation](https://pandas.pydata.org/docs/)
- [pandas API Reference](https://pandas.pydata.org/docs/reference/index.html)
- [matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Bitstamp USD Data](https://www.kaggle.com/datasets/tenebrist/bitstamp-usd-2012-2020)
