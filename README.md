# DLH — Machine Learning

A structured curriculum building the mathematical and data engineering foundations for machine learning.

---

## Directory Structure

```
dlh-machine_learning/
├── math/                        # Mathematical foundations
│   ├── linear_algebra/          # Matrix operations: Python lists → NumPy
│   │   ├── 0-slice_me_up.py     through 14-saddle_up.py
│   │   ├── 100-slice_like_a_ninja.py through 102-squashed_like_sardines.py
│   │   └── README.md
│   └── README.md
├── pipeline/                    # Data engineering
│   ├── databases/               # SQL: creation, CRUD, joins, aggregates, triggers
│   │   ├── 0-create_database_if_missing.sql through 18-valid_email.sql
│   │   ├── hbtn_0d_tvshows.sql, hbtn_0d_tvshows_rate.sql
│   │   ├── metal_bands.sql, temperatures.sql
│   │   └── README.md
│   └── README.md
├── my-venv/                     # Python virtual environment
└── README.md
```

---

## Quick Reference

| Track | Module | Topics | Tasks |
|-------|--------|--------|-------|
| **Math** | [Linear Algebra](math/linear_algebra/) | Slicing, shape, transpose, element-wise ops, concat, matrix multiply, NumPy, n-D generalization | 19 |
| **Pipeline** | [Databases](pipeline/databases/) | DDL, CRUD, WHERE, ORDER BY, GROUP BY, JOINS, aggregates, constraints, triggers | 18 (+4 schemas) |

---

## Learning Progression

### Math Track
1. **Python Slicing** → 2. **Manual Matrix Ops** (nested loops) → 3. **NumPy Vectorization** → 4. **N-Dimensional Generalization**

### Pipeline Track
1. **Foundation** (CREATE) → 2. **CRUD** → 3. **Filtering/Sorting** → 4. **Joins** → 5. **Constraints** → 6. **Real-World Data** → 7. **Triggers**

---

## Setup

```bash
cd dlh-machine_learning
source my-venv/bin/activate
pip install numpy
```

---

## Resources

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [MySQL Documentation](https://dev.mysql.com/doc/)
- [Python Official Documentation](https://docs.python.org/3/)
