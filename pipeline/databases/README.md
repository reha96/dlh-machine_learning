# Pipeline — Databases

A progressive journey through SQL and MongoDB — from basic table creation and CRUD operations to stored procedures, user-defined functions, indexes, views, and NoSQL document operations with Python/PyMongo.

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
| 10 | Create stored procedures with variables, conditionals, and input parameters |
| 11 | Define user-defined functions (UDFs) for reusable safe computations |
| 12 | Optimize queries with prefix and composite indexes |
| 13 | Create views for dynamic, reusable filtered datasets |
| 14 | Compute weighted averages with JOIN in stored procedures |
| 15 | Interact with MongoDB via the shell: `show dbs`, `use`, `find`, `insert`, `update`, `delete` |
| 16 | Perform CRUD operations programmatically with PyMongo in Python |
| 17 | Build MongoDB aggregation pipelines (`$group`, `$sort`, `$limit`) and regex queries (`$regex`) |

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

### Task 19 — Stored Procedure: Add Bonus (`19-bonus.sql`)

**Challenge:** Create a reusable stored procedure that adds a correction for a student, auto-creating the project if it doesn't exist — introducing procedural SQL with variables and conditional logic.

**Approach:** `CREATE PROCEDURE AddBonus(IN p_user_id INT, IN p_project_name VARCHAR(255), IN p_score INT)` — declares a local variable with `DECLARE`, checks if the project exists with `SELECT id INTO`, and conditionally inserts a new project with `IF ... THEN`. Finally inserts the correction.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `CREATE PROCEDURE name(IN param TYPE, ...)` | Define a stored procedure with input parameters |
| `DECLARE var_name TYPE` | Declare a local variable inside a procedure body |
| `SELECT col INTO var FROM table WHERE ...` | Store a query result into a variable |
| `IF var IS NULL THEN ... END IF` | Conditional logic inside a stored procedure |

> **Key takeaway:** Stored procedures encapsulate multi-step logic — check, branch, insert — into a single callable unit. `SELECT ... INTO` is how you capture query results in procedural SQL.

---

### Task 20 — Stored Procedure: Compute Average Score (`20-average_score.sql`)

**Challenge:** Compute a student's average score across all corrections and store it back into the `users` table — automating derived data with a stored procedure.

**Approach:** `CREATE PROCEDURE ComputeAverageScoreForUser(IN p_user_id INT)` — uses `SELECT AVG(score) INTO` to compute the average, then `UPDATE users SET average_score = IFNULL(var, 0)` to persist it, gracefully handling students with no corrections.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `AVG()` in a stored procedure | Compute aggregates inside procedural SQL |
| `UPDATE ... SET col = value WHERE id = param` | Persist computed results back to the database |
| `IFNULL(expr, default)` | Return a default value (0) when the expression is NULL |

> **Key takeaway:** Stored procedures can bridge the gap between raw data and derived values — compute once, store for fast retrieval. `IFNULL` prevents NULL from propagating into your computed columns.

---

### Task 21 — User-Defined Function: Safe Division (`21-div.sql`)

**Challenge:** Create a reusable function that performs safe division — returning 0 instead of raising a divide-by-zero error.

**Approach:** `CREATE FUNCTION SafeDiv(first_num INT, second_num INT) RETURNS FLOAT DETERMINISTIC NO SQL` — checks `IF second_num = 0 THEN RETURN 0`, otherwise returns `first_num / second_num`. The function is marked `DETERMINISTIC` (same inputs → same output) and `NO SQL` (no data reads/writes).

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `CREATE FUNCTION name(params) RETURNS TYPE` | Define a user-defined function (UDF) |
| `DETERMINISTIC` | Declare the function always returns the same result for the same inputs |
| `NO SQL` | Declare the function does not read or write database data |
| `RETURN value` inside IF/ELSE | Return different values based on conditions |

> **Key takeaway:** UDFs differ from procedures — they return a value and can be used inline in `SELECT` statements. `SafeDiv` encapsulates a business rule (no division by zero) in one reusable place.

---

### Task 100 — Index on First Letter (`100-index_my_names.sql`)

**Challenge:** Speed up name lookups by creating an index on only the first letter of the `name` column — introducing prefix/partial indexes.

**Approach:** `CREATE INDEX idx_name_first ON names (name(1))` — indexes just the first character of `name`, dramatically reducing index size while still accelerating prefix searches.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `CREATE INDEX index_name ON table (col(N))` | Create a prefix index on the first N characters of a column |
| `name(1)` — single-character index | Index only what you query — minimal storage, maximal speed for prefix searches |

> **Key takeaway:** You don't need to index the entire column. A prefix index on `name(1)` is perfect for queries like `WHERE name LIKE 'A%'` — tiny index, huge speedup.

---

### Task 101 — Composite Index (`101-index_name_score.sql`)

**Challenge:** Optimize queries that filter by both first letter of name AND score — introducing multi-column composite indexes.

**Approach:** `CREATE INDEX idx_name_first_score ON names (name(1), score)` — the index covers both columns in order, so queries filtering on `name(1)` alone OR `name(1) + score` both benefit.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `CREATE INDEX ON table (col1, col2)` | Create a composite (multi-column) index |
| Column order matters | Leftmost prefix rule — index on `(A, B)` helps queries on `A` alone, not `B` alone |

> **Key takeaway:** Composite indexes cover multiple columns. Order matters: `(name(1), score)` helps `WHERE name LIKE 'A%' AND score > 80`, but NOT `WHERE score > 80` alone.

---

### Task 102 — View: Students Needing Meeting (`102-need_meeting.sql`)

**Challenge:** Create a dynamically-filtered view of students with low scores and stale/no meetings — introducing `CREATE VIEW` with date arithmetic.

**Approach:** `CREATE VIEW need_meeting AS SELECT name FROM students WHERE score < 80 AND (last_meeting IS NULL OR last_meeting < CURDATE() - INTERVAL 1 MONTH)` — the view encapsulates the filtering logic and always returns current results.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `CREATE VIEW name AS SELECT ...` | Define a virtual table based on a query |
| `CURDATE()` | Get the current date (no time component) |
| `INTERVAL 1 MONTH` | Date arithmetic — subtract one month from a date |
| `IS NULL` OR date comparison | Handle both "never met" and "met too long ago" cases |

> **Key takeaway:** Views are saved queries — they don't store data, they rerun the query each time. `CURDATE() - INTERVAL 1 MONTH` is how you express "more than a month ago" in SQL.

---

### Task 103 — Stored Procedure: Weighted Average Score (`103-average_weighted_score.sql`)

**Challenge:** Compute a student's weighted average score — where each project has a different weight — and store the result. Introducing `JOIN` inside a stored procedure with weighted aggregation.

**Approach:** `CREATE PROCEDURE ComputeAverageWeightedScoreForUser(IN p_user_id INT)` — uses `SELECT SUM(score * projects.weight) / SUM(projects.weight) INTO` from `corrections JOIN projects` to compute the weighted average, then updates the user record.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `SUM(score * weight) / SUM(weight)` | Weighted average formula: sum of (value × weight) divided by sum of weights |
| `JOIN` inside a stored procedure | Combine data from multiple tables in procedural SQL |
| `INTO` with aggregate expression | Store a computed aggregate directly into a variable |

> **Key takeaway:** A weighted average accounts for unequal importance — a project worth 5× counts 5× more. The formula $\frac{\sum(\text{score} \times \text{weight})}{\sum \text{weight}}$ handles any weight distribution.

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
| 19 | `CREATE PROCEDURE`, `IN` params, `DECLARE`, `SELECT INTO`, `IF` | Stored Procedures |
| 20 | `AVG()` in procedure, `UPDATE` with `IFNULL` | Stored Procedures |
| 21 | `CREATE FUNCTION`, `RETURNS`, `DETERMINISTIC`, `NO SQL` | User-Defined Functions |
| 100 | `CREATE INDEX ON name(1)` — prefix index | Indexes |
| 101 | Composite index `(name(1), score)` — multi-column | Indexes |
| 102 | `CREATE VIEW`, `CURDATE()`, `INTERVAL 1 MONTH` | Views |
| 103 | `SUM(score*weight)/SUM(weight)`, `JOIN` in procedure | Stored Procedures |

---

## MongoDB Shell (Tasks 22–29)

Each task below highlights the **unique challenge** it posed and the **new technique** introduced — techniques from MySQL tasks are not repeated.

---

### Task 22 — List Databases (`22-list_databases`)

**Challenge:** Discover what databases exist on the MongoDB server — the first interaction with a new NoSQL environment.

**Approach:** `show dbs` — MongoDB shell's equivalent of `SHOW DATABASES` in MySQL. Returns all database names and their sizes.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `show dbs` | List all databases on the MongoDB server |
| MongoDB shell (`mongosh`) | Interactive JavaScript-based shell for MongoDB operations |

> **Key takeaway:** `show dbs` is your first command in MongoDB — it confirms the server is reachable and shows available databases. Unlike MySQL, databases are lazily created on first insert.

---

### Task 23 — Select/Create Database (`23-use_or_create_database`)

**Challenge:** Switch to (or implicitly create) a database — MongoDB has no explicit `CREATE DATABASE`.

**Approach:** `use my_db` — MongoDB creates the database automatically when you first insert data into it. The `use` command switches context without requiring the database to already exist.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `use db_name` | Switch to a database (creates it lazily on first write) |
| Implicit database creation | No `CREATE DATABASE` needed — databases spring into existence on first insert |

> **Key takeaway:** In MongoDB, `use` switches context; the database isn't physically created until you insert a document. This is fundamentally different from MySQL's explicit `CREATE DATABASE`.

---

### Task 24 — Insert Document (`24-insert`)

**Challenge:** Add data to a MongoDB collection — introducing document insertion with JSON-like syntax.

**Approach:** `db.school.insert({ name: "Holberton school" })` — insert a BSON document (JavaScript object) into the `school` collection. Collections are created implicitly on first insert.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `db.collection.insert({...})` | Insert a new document into a collection |
| JSON/BSON document syntax | Key-value pairs in `{ field: value }` format |
| Implicit collection creation | Collections are created automatically on first `insert()` |

> **Key takeaway:** MongoDB stores JSON-like documents. `insert()` adds a document to a collection — both the collection and database are created implicitly if they don't exist.

---

### Task 25 — Find All Documents (`25-all`)

**Challenge:** Retrieve all documents from a collection — the MongoDB equivalent of `SELECT *`.

**Approach:** `db.school.find()` — without arguments, returns every document in the collection. Outputs documents in readable JSON format.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `db.collection.find()` | Retrieve all documents from a collection |
| Cursor output | `find()` returns a cursor that the shell iterates and displays |

> **Key takeaway:** `find()` with no arguments returns all documents — MongoDB's `SELECT *`. The shell automatically iterates the cursor and pretty-prints results.

---

### Task 26 — Find with Filter (`26-match`)

**Challenge:** Retrieve only documents matching a specific condition — introducing query filters.

**Approach:** `db.school.find({ name: "Holberton school" })` — pass a query document to filter results. Only documents whose `name` field matches the value are returned.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `find({ field: value })` | Filter documents by exact field match |
| Query document | MongoDB uses `{key: value}` objects as query predicates |

> **Key takeaway:** MongoDB queries are themselves JSON-like documents. `find({name: "X"})` is equivalent to `WHERE name = 'X'` in SQL — the filter is declarative.

---

### Task 27 — Count Documents (`27-count`)

**Challenge:** Count how many documents exist in a collection — introducing document counting.

**Approach:** `db.school.count()` — returns the total number of documents. Can also accept a filter: `db.school.count({name: "Holberton school"})`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `db.collection.count()` | Return the total number of documents in a collection |
| `count({filter})` | Count only documents matching a query filter |

> **Key takeaway:** `count()` is MongoDB's `COUNT(*)`. It's a simple, fast way to check collection size or count matching documents.

---

### Task 28 — Update Documents (`28-update`)

**Challenge:** Modify existing documents by adding new fields — introducing the `$set` operator and `multi` option.

**Approach:** `db.school.update({ name: "Holberton school" }, { $set: { address: "972 Mission street" } }, { multi: true })` — match documents by `name`, then add an `address` field via `$set`. `multi: true` updates all matching documents, not just the first.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `db.collection.update(query, update, options)` | Modify documents matching a query |
| `$set` operator | Add or modify specific fields without replacing the entire document |
| `{ multi: true }` | Update ALL matching documents (default updates only the first) |

> **Key takeaway:** `$set` modifies only specified fields — the rest of the document is untouched. Always use `{ multi: true }` when you intend to update multiple documents; without it, only the first match is changed.

---

### Task 29 — Delete Documents (`29-delete`)

**Challenge:** Remove documents from a collection — introducing bulk deletion.

**Approach:** `db.school.deleteMany({ name: "Holberton school" })` — delete all documents matching the filter. `deleteMany()` removes every match; `deleteOne()` would remove only the first.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `db.collection.deleteMany({filter})` | Delete all documents matching a query filter |
| `deleteOne()` vs `deleteMany()` | `deleteOne` removes first match; `deleteMany` removes all matches |

> **Key takeaway:** `deleteMany()` is MongoDB's `DELETE FROM ... WHERE ...` — it removes all matching documents in one operation. Use `deleteOne()` when you want to limit removal to a single document.

---

### Task 104 — Find with Regex (`104-find`)

**Challenge:** Find all documents whose `name` field starts with a specific prefix — introducing MongoDB's `$regex` operator for pattern matching.

**Approach:** `db.school.find({ name: { $regex: "^Holberton" } })` — the `$regex` operator applies a regular expression to the `name` field. `^` anchors the match to the start of the string, so only names beginning with "Holberton" match.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `{ field: { $regex: "pattern" } }` | Apply a regular expression filter to a field |
| `^` anchor in regex | Match only at the start of the string (prefix matching) |
| MongoDB regex vs SQL `LIKE` | `$regex: "^Holberton"` ≈ `WHERE name LIKE 'Holberton%'` in SQL |

> **Key takeaway:** MongoDB's `$regex` is the NoSQL equivalent of SQL's `LIKE` — it enables pattern-based queries. The `^` anchor makes it a prefix search, which can leverage indexes for performance.

---

## Python & PyMongo (Tasks 30–34)

Transitioning from the MongoDB shell to programmatic database access with Python's `pymongo` driver.

---

### Task 30 — List All Documents in Python (`30-all.py`)

**Challenge:** Retrieve all documents from a MongoDB collection and return them as a Python list — bridging the shell-to-Python gap.

**Approach:** `list(mongo_collection.find())` — `find()` returns a Cursor; wrapping it in `list()` materializes all documents into a Python list of dicts.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `pymongo` collection object | Python driver interface matching shell commands |
| `mongo_collection.find()` | Return a Cursor over all documents in the collection |
| `list(cursor)` | Materialize a Cursor into a Python list of dictionaries |

> **Key takeaway:** PyMongo's API mirrors the shell — `collection.find()` is equivalent to `db.collection.find()`. Wrap the cursor in `list()` to get all documents at once.

---

### Task 31 — Insert with Keyword Arguments (`31-insert_school.py`)

**Challenge:** Accept arbitrary keyword arguments and insert them as a MongoDB document — introducing dynamic document creation.

**Approach:** `mongo_collection.insert(kwargs)` — Python's `**kwargs` captures all keyword arguments as a dictionary, then `insert()` stores it directly as a MongoDB document.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `**kwargs` parameter | Capture arbitrary keyword arguments as a dictionary |
| `insert(document)` | Insert a Python dict as a MongoDB document (returns `_id`) |

> **Key takeaway:** `**kwargs` + `insert()` lets you pass any field-value pairs from Python directly into MongoDB — the Python dict maps 1:1 to a BSON document.

---

### Task 32 — Update Many with `$set` (`32-update_topics.py`)

**Challenge:** Update the `topics` field for ALL schools matching a given name — translating the shell update pattern to Python.

**Approach:** `mongo_collection.update_many({'name': name}, {'$set': {'topics': topics}})` — filter by `name`, then use `$set` to add/overwrite the `topics` field. `update_many()` ensures all matches are updated.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `update_many(filter, update)` | Update all documents matching the filter |
| `{'$set': {field: value}}` in Python | Same `$set` operator, expressed as a Python dict |

> **Key takeaway:** PyMongo update operations use the same MongoDB operators (`$set`, `$inc`, etc.) — just expressed as Python dicts. `update_many()` is the safe default; `update_one()` only touches the first match.

---

### Task 33 — Find by Array Field (`33-schools_by_topic.py`)

**Challenge:** Find all schools whose `topics` array contains a specific value — introducing array field matching in MongoDB.

**Approach:** `mongo_collection.find({'topics': topic})` — when the field is an array, MongoDB automatically matches documents where ANY element equals the query value.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| Array field matching | `find({array_field: value})` matches if any array element equals `value` |
| Implicit `$in` behavior | MongoDB treats scalar-to-array matching as "contains" |

> **Key takeaway:** MongoDB intelligently matches array fields — `find({'topics': 'math'})` finds documents where `topics` is an array containing `'math'`. No special syntax needed.

---

### Task 34 — Nginx Log Analysis (`34-log_stats.py`)

**Challenge:** Connect to a real MongoDB instance and analyze Nginx access logs — introducing `MongoClient`, database/collection access, and `count_documents()`.

**Approach:** `MongoClient('mongodb://127.0.0.1:27017')` connects to the local server. Then `client.logs.nginx` navigates to the `nginx` collection in the `logs` database. `count_documents({})` counts total logs; `count_documents({'method': 'GET'})` counts per HTTP method.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `MongoClient(uri)` | Establish a connection to a MongoDB server |
| `client.db_name.collection_name` | Attribute-style access to database and collection |
| `count_documents({filter})` | Count documents matching a filter (modern replacement for `count()`) |
| Real-world data analysis | Apply MongoDB queries to production Nginx log data |

> **Key takeaway:** `MongoClient` is the entry point — it connects to MongoDB, and from there you navigate databases and collections with dot notation. `count_documents()` is the modern, preferred way to count.

---

## Python & PyMongo — Aggregation (Tasks 105–106)

Advanced data analysis using MongoDB's aggregation pipeline — grouping, sorting, and limiting results.

---

### Task 105 — Students Sorted by Average Score (`105-students.py`)

**Challenge:** Compute each student's average score from a nested `topics` array and sort them by that average — introducing the aggregation pipeline.

**Approach:** A two-stage pipeline: `$addFields` creates a new `averageScore` field by averaging `$topics.score` with `$avg`, then `$sort` orders by `averageScore` descending. `aggregate(pipeline)` executes and returns a cursor.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `aggregate([pipeline_stages])` | Execute a sequence of data transformations |
| `$addFields` stage | Add computed fields to each document |
| `$avg: '$nested.field'` | Compute the average of an array field's values |
| `$sort: {field: -1}` | Sort results in descending order (`-1`) |

> **Key takeaway:** The aggregation pipeline processes documents through stages in sequence — each stage transforms the data flowing from the previous one. `$addFields` enriches documents; `$sort` reorders them. Together they form a composable data processing chain.

---

### Task 106 — Top 10 IPs (`106-log_stats.py`)

**Challenge:** Extend the Nginx log analysis (Task 34) to find the 10 most frequent IP addresses — introducing `$group`, `$sum`, and `$limit`.

**Approach:** A three-stage pipeline appended to the existing script: `$group` clusters documents by `$ip` and counts each with `$sum: 1`, `$sort` orders by count descending, `$limit: 10` keeps only the top 10.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `$group` stage | Group documents by a field, computing per-group aggregates |
| `_id: '$field'` in `$group` | Specify the grouping key (field to group by) |
| `$sum: 1` | Count documents per group (1 per document, summed) |
| `$limit: N` | Restrict output to the first N documents |

> **Key takeaway:** `$group` is MongoDB's `GROUP BY` — `_id` defines the bucket, accumulator expressions (`$sum`, `$avg`, `$max`) compute per-group values. `$limit` caps results, making it perfect for "top N" queries.

---

## Technique Inventory — MongoDB

| Task | New technique summarized | Category |
|------|--------------------------|----------|
| 22 | `show dbs` — list all databases | MongoDB Shell |
| 23 | `use db_name` — implicit DB creation | MongoDB Shell |
| 24 | `db.collection.insert({...})` — document insertion | MongoDB CRUD |
| 25 | `db.collection.find()` — retrieve all documents | MongoDB CRUD |
| 26 | `find({field: value})` — query with filter | MongoDB Queries |
| 27 | `db.collection.count()` — document counting | MongoDB Queries |
| 28 | `$set` operator, `update()`, `multi: true` | MongoDB Updates |
| 29 | `deleteMany()` — bulk document deletion | MongoDB CRUD |
| 30 | `list(mongo_collection.find())` — materialize cursor to Python list | PyMongo CRUD |
| 31 | `insert(kwargs)`, `**kwargs` — dynamic insert | PyMongo CRUD |
| 32 | `update_many()` with `$set` in Python | PyMongo Updates |
| 33 | Array field matching in `find()` | PyMongo Queries |
| 34 | `MongoClient`, `count_documents()` — real-world analysis | PyMongo Connection |
| 105 | `aggregate()`, `$addFields`, `$avg`, `$sort` | Aggregation Pipeline |
| 106 | `$group`, `$sum`, `$limit` — top-N grouping | Aggregation Pipeline |
| 104 | `$regex: "^prefix"` — pattern-matching query | MongoDB Queries |

---

## Resources

### MySQL
- [MySQL CREATE DATABASE — Official Docs](https://dev.mysql.com/doc/refman/8.0/en/create-database.html)
- [MySQL CREATE TABLE — Official Docs](https://dev.mysql.com/doc/refman/8.0/en/create-table.html)
- [MySQL JOIN Syntax](https://dev.mysql.com/doc/refman/8.0/en/join.html)
- [MySQL Aggregate Functions](https://dev.mysql.com/doc/refman/8.0/en/aggregate-functions.html)
- [MySQL Triggers — Official Docs](https://dev.mysql.com/doc/refman/8.0/en/trigger-syntax.html)
- [MySQL ENUM — Official Docs](https://dev.mysql.com/doc/refman/8.0/en/enum.html)
- [SQL COALESCE — Official Docs](https://dev.mysql.com/doc/refman/8.0/en/comparison-operators.html#function_coalesce)

### MongoDB
- [MongoDB Shell (mongosh) — Official Docs](https://www.mongodb.com/docs/mongodb-shell/)
- [MongoDB CRUD Operations](https://www.mongodb.com/docs/manual/crud/)
- [MongoDB Query Documents](https://www.mongodb.com/docs/manual/tutorial/query-documents/)
- [MongoDB Update Operators (`$set`)](https://www.mongodb.com/docs/manual/reference/operator/update/set/)
- [MongoDB Aggregation Pipeline](https://www.mongodb.com/docs/manual/aggregation/)
- [PyMongo Documentation](https://pymongo.readthedocs.io/)