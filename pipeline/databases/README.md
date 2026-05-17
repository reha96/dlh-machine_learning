# Pipeline — Databases

A progressive journey through SQL — from basic table creation and CRUD operations to multi-table joins, aggregations, constraints, and automated triggers.

---

## Learning Objectives

| # | Concept |
|---|---------|
| 1 | Create databases and tables with proper data types and constraints |
| 2 | Perform CRUD operations: `SELECT`, `INSERT`, `UPDATE`, `DELETE` |
| 3 | Filter and sort data with `WHERE`, `ORDER BY`, and `LIKE` |
| 4 | Aggregate data across groups with `GROUP BY`, `AVG`, `MAX`, `COUNT`, `SUM` |
| 5 | Join multiple tables with `INNER JOIN`, `LEFT JOIN`, and chained joins |
| 6 | Enforce data integrity with `NOT NULL`, `UNIQUE`, `PRIMARY KEY`, `AUTO_INCREMENT`, `ENUM` |
| 7 | Handle null values and computed columns with `COALESCE` and arithmetic |
| 8 | Automate database logic with `CREATE TRIGGER` (BEFORE/AFTER, INSERT/UPDATE) |
| 9 | Design schemas for many-to-many relationships and time-series data |

---

## Task-by-Task Reference

Each task below highlights the **unique challenge** it posed and the **new technique** introduced — techniques from earlier tasks are not repeated.

---

### Task 0 — Create Database (`0-create_database_if_missing.sql`)

**Challenge:** Set up the foundational container for all subsequent work — the database itself — with idempotent execution.

**Approach:** `CREATE DATABASE IF NOT EXISTS db_0` — safely re-runnable without errors.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `CREATE DATABASE IF NOT EXISTS` | Create a new database only if it doesn't already exist |
| Idempotent DDL | Safe to run multiple times — no side effects on re-execution |

> **Key takeaway:** Every SQL workflow starts with database creation. `IF NOT EXISTS` makes scripts safe for repeated execution.

---

### Task 1 — Create Table (`1-first_table.sql`)

**Challenge:** Define a table structure with typed columns — introducing DDL and data type selection.

**Approach:** `CREATE TABLE IF NOT EXISTS first_table (id INT, name VARCHAR(256))` — two columns with specific types, backtick-quoted identifiers.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `CREATE TABLE IF NOT EXISTS` | Define a table structure with idempotent execution |
| `INT` data type | Integer column for numeric IDs |
| `VARCHAR(256)` data type | Variable-length string up to 256 characters |
| Backtick-quoted identifiers | Escape table/column names that may conflict with reserved words |

> **Key takeaway:** Every table column needs an explicit data type. Choose `INT` for numbers, `VARCHAR(N)` for strings — the type constrains what data can be stored.

---

### Task 2 — List All Rows (`2-list_values.sql`)

**Challenge:** Retrieve the entire contents of a table — the simplest possible read operation.

**Approach:** `SELECT * FROM first_table` — the wildcard `*` pulls every column.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `SELECT * FROM table` | Retrieve all columns and all rows |
| `FROM` clause | Specify which table to query |

> **Key takeaway:** `SELECT *` is the simplest read operation — it returns every row and every column from the specified table.

---

### Task 3 — Insert Row (`3-insert_value.sql`)

**Challenge:** Add data to the table — introducing the write side of CRUD.

**Approach:** `INSERT INTO first_table (id, name) VALUES (89, 'Best School')` — explicit column naming makes the intent clear.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `INSERT INTO table (col1, col2) VALUES (v1, v2)` | Add a new row with specified column values |
| Explicit column specification | Makes insertion order-independent and self-documenting |

> **Key takeaway:** Named columns in `INSERT` make the operation explicit and maintainable — you're not relying on column order.

---

### Task 4 — Filter and Sort (`4-best_score.sql`)

**Challenge:** Retrieve only records meeting a condition, ordered by a specific column — introducing `WHERE` and `ORDER BY`.

**Approach:** `SELECT score, name FROM second_table WHERE score >= 10 ORDER BY score DESC` — filter first, then sort.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `WHERE condition` | Filter rows — keep only those matching the condition |
| `>=` comparison operator | Greater-than-or-equal filter |
| `ORDER BY column DESC` | Sort results in descending order |

> **Key takeaway:** `WHERE` filters rows before they're returned; `ORDER BY` sequences the surviving rows. Both operate on the result set, not the table.

---

### Task 5 — Average (`5-average.sql`)

**Challenge:** Compute a single aggregate value across all rows — introducing aggregate functions.

**Approach:** `SELECT AVG(score) AS average FROM second_table` — collapses all rows into one number.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `AVG(column)` | Compute the arithmetic mean of a numeric column |
| `AS alias` | Give the result column a readable name |

> **Key takeaway:** Aggregate functions like `AVG()` collapse multiple rows into a single computed value — the result is one row, not one per input row.

---

### Task 6 — Aggregate with Grouping (`6-avg_temperatures.sql`)

**Challenge:** Compute averages per city — introducing `GROUP BY` for partitioned aggregation.

**Approach:** `SELECT city, AVG(value) AS avg_temp FROM temperatures GROUP BY city ORDER BY avg_temp DESC` — group, aggregate within each group, then sort groups.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `GROUP BY column` | Partition rows into groups based on a column's values |
| Aggregate + GROUP BY | `AVG()` computes per-group, not global |
| `ORDER BY aggregate_alias DESC` | Sort groups by their aggregate value |

> **Key takeaway:** `GROUP BY` partitions data before aggregation — each group gets its own `AVG()`, `MAX()`, or `COUNT()`. Without it, aggregates apply globally.

---

### Task 7 — Maximum with Grouping (`7-max_state.sql`)

**Challenge:** Find the maximum temperature per state — applying `MAX()` in a grouped context.

**Approach:** `SELECT state, MAX(value) AS max_temp FROM temperatures GROUP BY state ORDER BY state` — same pattern as Task 6, with `MAX()` instead of `AVG()`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `MAX(column)` | Find the highest value in a column (or per group) |

> **Key takeaway:** `MAX()` finds the highest value per group — same `GROUP BY` pattern, different aggregate. The grouping logic is identical across all aggregates.

---

### Task 8 — Inner Join (`8-genre_id_by_show.sql`)

**Challenge:** Connect two related tables to list shows that have genres — introducing `INNER JOIN`.

**Approach:** `SELECT tv_shows.title, tv_show_genres.genre_id FROM tv_shows INNER JOIN tv_show_genres ON tv_shows.id = tv_show_genres.show_id ORDER BY title, genre_id`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `INNER JOIN table ON condition` | Combine rows from two tables where the condition matches |
| `table.column` qualified names | Disambiguate columns when multiple tables share names |
| `ORDER BY col1, col2` | Multi-column sort (primary by title, secondary by genre_id) |

> **Key takeaway:** `INNER JOIN` returns only rows where a match exists in BOTH tables. If a show has no genre, it's excluded from the result.

---

### Task 9 — Left Join: Missing Genres (`9-no_genre.sql`)

**Challenge:** List shows that have NO linked genre — the anti-join pattern using `LEFT JOIN` + `IS NULL`.

**Approach:** `SELECT tv_shows.title, tv_show_genres.genre_id FROM tv_shows LEFT JOIN tv_show_genres ON tv_shows.id = tv_show_genres.show_id WHERE tv_show_genres.genre_id IS NULL ORDER BY title`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `LEFT JOIN` | Keep ALL rows from the left table — add right-table data where a match exists |
| `WHERE column IS NULL` | Find rows where the join produced no match |
| Anti-join pattern | `LEFT JOIN + IS NULL` = "find records in A with no match in B" |

> **Key takeaway:** `LEFT JOIN` preserves every row from the left table. When there's no match, right-table columns are `NULL` — filter on that to find orphans.

---

### Task 10 — Count with Grouping (`10-count_shows_by_genre.sql`)

**Challenge:** Count how many shows belong to each genre — introducing `COUNT()` with joins and grouping.

**Approach:** `SELECT tv_genres.name AS genre, COUNT(tv_show_genres.genre_id) AS number_of_shows FROM tv_genres LEFT JOIN tv_show_genres ON tv_genres.id = tv_show_genres.genre_id GROUP BY tv_genres.name ORDER BY number_of_shows DESC`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `COUNT(column)` | Count non-NULL values — with `LEFT JOIN`, counts only matched rows |
| `COUNT` + `LEFT JOIN` pattern | Genres with 0 shows appear as 0 in the count |
| `GROUP BY text_column` | Group by a name/string, not just IDs |

> **Key takeaway:** `COUNT()` with `LEFT JOIN` ensures even empty categories appear (with count 0). `INNER JOIN` + `COUNT()` would silently drop them.

---

### Task 11 — Sum Ratings (`11-rating_shows.sql`)

**Challenge:** Sum ratings for each show — introducing `SUM()` aggregate.

**Approach:** `SELECT tv_shows.title, SUM(tv_show_ratings.rate) AS rating FROM tv_shows LEFT JOIN tv_show_ratings ON tv_shows.id = tv_show_ratings.show_id GROUP BY tv_shows.title ORDER BY rating DESC`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `SUM(column)` | Total all values in a numeric column (or per group) |
| `LEFT JOIN` for optional ratings | Shows with no ratings still appear (SUM = NULL or 0) |

> **Key takeaway:** `SUM()` totals numeric values. With `LEFT JOIN`, shows without ratings return `NULL` — use `COALESCE(SUM(...), 0)` if you want 0 instead.

---

### Task 12 — Sum with Chained Joins (`12-rating_genres.sql`)

**Challenge:** Sum ratings by genre — navigating three tables in one query with chained `LEFT JOIN`s.

**Approach:** `SELECT tv_genres.name, SUM(tv_show_ratings.rate) AS rating FROM tv_genres LEFT JOIN tv_show_genres ON tv_genres.id = tv_show_genres.genre_id LEFT JOIN tv_show_ratings ON tv_show_genres.show_id = tv_show_ratings.show_id GROUP BY tv_genres.name ORDER BY rating DESC`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Multiple consecutive `LEFT JOIN`s | Chain through intermediate tables to reach the target data |
| Junction table traversal | `genres → show_genres → show_ratings` — the junction table bridges the relationship |

> **Key takeaway:** Chain multiple joins to traverse complex relationships. Each join adds one more hop through the schema — the query grows linearly with the path length.

---

### Task 13 — Unique Constraints (`13-uniq_users.sql`)

**Challenge:** Create a `users` table that prevents duplicate emails — introducing data integrity constraints.

**Approach:** `CREATE TABLE IF NOT EXISTS users (id INT NOT NULL AUTO_INCREMENT, email VARCHAR(256) NOT NULL UNIQUE, name VARCHAR(256), PRIMARY KEY (id))`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `AUTO_INCREMENT` | Automatically generate sequential integer IDs |
| `NOT NULL` | Reject INSERTs that omit this column |
| `UNIQUE` | Reject INSERTs that duplicate an existing value |
| `PRIMARY KEY (col)` | Uniquely identify each row — implies NOT NULL + UNIQUE |

> **Key takeaway:** Constraints like `NOT NULL`, `UNIQUE`, and `PRIMARY KEY` enforce data integrity at the database level — bad data is rejected before it enters the table.

---

### Task 14 — ENUM Constraint (`14-country_users.sql`)

**Challenge:** Extend the `users` table with a country column restricted to three specific values — introducing enumerated types.

**Approach:** `CREATE TABLE ... country ENUM('US', 'CO', 'TN') NOT NULL DEFAULT 'US'` — only these three values are accepted.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `ENUM('val1', 'val2', 'val3')` | Restrict a column to a predefined set of string values |
| `DEFAULT value` | Provide a fallback value when none is specified in INSERT |

> **Key takeaway:** `ENUM` acts as a check constraint on strings — the database rejects any value not in the list. Use it for columns with a fixed, known set of valid options.

---

### Task 15 — Real-World Aggregation (`15-fans.sql`)

**Challenge:** Rank band origins by total fans from a real dataset — applying grouping and summing to non-trivial data.

**Approach:** `SELECT origin, SUM(fans) AS nb_fans FROM metal_bands GROUP BY origin ORDER BY nb_fans DESC` — group by text column, sum numeric.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Grouping by text/string column | `GROUP BY origin` works on VARCHAR just like INT |
| Real dataset context | Apply SQL patterns to meaningful, large-scale data |

> **Key takeaway:** `GROUP BY` and `SUM()` work with any data type — string columns for grouping, numeric columns for aggregation. The pattern is universal.

---

### Task 16 — Pattern Matching and COALESCE (`16-glam_rock.sql`)

**Challenge:** Find Glam rock bands, compute their lifespan, and handle active bands (no split year) — introducing `LIKE` and `COALESCE`.

**Approach:** `SELECT band_name, COALESCE(split, 2020) - formed AS lifespan FROM metal_bands WHERE style LIKE '%Glam rock%' ORDER BY lifespan DESC`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `LIKE '%pattern%'` | Pattern matching with `%` wildcards (0+ characters) |
| `COALESCE(val, default)` | Return the first non-NULL value — treat NULL `split` as "still active" |
| Arithmetic in SELECT | Compute derived columns (`split - formed`) directly in the query |

> **Key takeaway:** `COALESCE()` provides fallback values for NULL — use it to handle missing data gracefully. Arithmetic in SELECT creates computed columns on the fly.

---

### Task 17 — AFTER INSERT Trigger (`17-store.sql`)

**Challenge:** Automatically decrement item quantity when an order is placed — introducing database triggers for reactive logic.

**Approach:** `CREATE TRIGGER decrease_quantity AFTER INSERT ON orders FOR EACH ROW UPDATE items SET quantity = quantity - NEW.number WHERE name = NEW.item_name`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `CREATE TRIGGER name timing event ON table` | Define automated logic that fires on data changes |
| `AFTER INSERT` | Execute the trigger body after a row is inserted |
| `FOR EACH ROW` | Run the trigger once per affected row |
| `NEW.column` | Reference the newly inserted row's column values |
| `DELIMITER //` | Change statement terminator to allow `;` inside trigger body |

> **Key takeaway:** Triggers automate reactive operations — when an order is placed, inventory updates itself. `NEW` refers to the row being inserted; `OLD` refers to the row being updated/deleted.

---

### Task 18 — BEFORE UPDATE Trigger (`18-valid_email.sql`)

**Challenge:** Reset the `valid_email` flag whenever a user's email address changes — introducing conditional trigger logic with `BEFORE UPDATE`.

**Approach:** `CREATE TRIGGER reset_valid_email BEFORE UPDATE ON users FOR EACH ROW IF OLD.email != NEW.email THEN SET NEW.valid_email = 0; END IF`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `BEFORE UPDATE` | Fire the trigger BEFORE the row is written — can modify `NEW` values |
| `IF condition THEN ... END IF` | Conditional logic inside a trigger body |
| `OLD.column` vs `NEW.column` | Compare old and new values to detect changes |
| `SET NEW.column = value` | Modify the incoming row before it's saved |

> **Key takeaway:** `BEFORE UPDATE` triggers can inspect and modify data before it hits the table. Compare `OLD` vs `NEW` to detect changes, then adjust `NEW` to enforce business rules.

---

## Schema Reference

### `hbtn_0d_tvshows.sql` — TV Shows + Genres (Many-to-Many)

| Table | Columns | Purpose |
|-------|---------|---------|
| `tv_genres` | `id` (PK), `name` | Genre catalog |
| `tv_shows` | `id` (PK), `title` | Show catalog |
| `tv_show_genres` | `show_id` (FK), `genre_id` (FK) | Junction table — links shows to genres |

**Pattern:** Junction table with composite foreign keys enables many-to-many relationships.

### `hbtn_0d_tvshows_rate.sql` — TV Shows + Genres + Ratings

Adds `tv_show_ratings` (show_id, rate) — a fact table allowing multiple ratings per show.

### `metal_bands.sql` — Bands Dataset

Single flat table: `metal_bands` (id, band_name, fans, formed, origin, split, style). Denormalized for simple queries. `YEAR` data type for formed/split. `split` is NULL for active bands.

### `temperatures.sql` — Weather Time-Series

Flat table: `temperatures` (city, state, year, month, value). Designed for efficient `GROUP BY` queries on city and time dimensions.

---

## Technique Inventory

| Task | New technique summarized | Category |
|------|--------------------------|----------|
| 0 | `CREATE DATABASE IF NOT EXISTS` | DDL — Database |
| 1 | `CREATE TABLE`, `INT`, `VARCHAR(N)` | DDL — Tables |
| 2 | `SELECT * FROM table` | CRUD — Read |
| 3 | `INSERT INTO ... VALUES` | CRUD — Create |
| 4 | `WHERE`, `ORDER BY DESC` | Filtering & Sorting |
| 5 | `AVG()`, `AS` aliasing | Aggregates |
| 6 | `GROUP BY`, aggregate + grouping | Grouped Aggregation |
| 7 | `MAX()` | Aggregates |
| 8 | `INNER JOIN ... ON` | Joins |
| 9 | `LEFT JOIN` + `IS NULL` (anti-join) | Joins |
| 10 | `COUNT()` with GROUP BY + LEFT JOIN | Aggregates + Joins |
| 11 | `SUM()` with GROUP BY | Aggregates |
| 12 | Multiple chained `LEFT JOIN`s | Joins |
| 13 | `AUTO_INCREMENT`, `NOT NULL`, `UNIQUE`, `PRIMARY KEY` | Constraints |
| 14 | `ENUM`, `DEFAULT` | Constraints |
| 15 | Real-world aggregation on text column | Aggregates |
| 16 | `LIKE '%pattern%'`, `COALESCE()`, arithmetic in SELECT | Pattern Matching & Nulls |
| 17 | `CREATE TRIGGER AFTER INSERT`, `NEW` reference | Triggers |
| 18 | `BEFORE UPDATE`, `IF` in trigger, `OLD` vs `NEW` | Triggers |

---

## Resources

- [MySQL CREATE DATABASE — Official Docs](https://dev.mysql.com/doc/refman/8.0/en/create-database.html)
- [MySQL CREATE TABLE — Official Docs](https://dev.mysql.com/doc/refman/8.0/en/create-table.html)
- [MySQL JOIN Syntax](https://dev.mysql.com/doc/refman/8.0/en/join.html)
- [MySQL Aggregate Functions](https://dev.mysql.com/doc/refman/8.0/en/aggregate-functions.html)
- [MySQL Triggers — Official Docs](https://dev.mysql.com/doc/refman/8.0/en/trigger-syntax.html)
- [MySQL ENUM — Official Docs](https://dev.mysql.com/doc/refman/8.0/en/enum.html)
- [SQL COALESCE — Official Docs](https://dev.mysql.com/doc/refman/8.0/en/comparison-operators.html#function_coalesce)