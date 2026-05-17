# Pipeline

Data engineering and database operations — SQL from table creation through complex joins, aggregations, constraints, and triggers.

## Modules

| Module | Description | Files |
|--------|-------------|-------|
| [Databases](databases/) | SQL: DDL, CRUD, filtering, grouping, joins, constraints, triggers, schema design | 18 tasks + 4 schema dumps |

## Learning Path

1. **Foundation** (Tasks 0–1): `CREATE DATABASE`, `CREATE TABLE`, data types
2. **CRUD Operations** (Tasks 2–3): `SELECT *`, `INSERT INTO`
3. **Filtering & Sorting** (Tasks 4–7): `WHERE`, `ORDER BY`, `GROUP BY`, `AVG`, `MAX`
4. **Joins** (Tasks 8–12): `INNER JOIN`, `LEFT JOIN`, chained joins, `COUNT`, `SUM`
5. **Constraints** (Tasks 13–14): `AUTO_INCREMENT`, `NOT NULL`, `UNIQUE`, `PRIMARY KEY`, `ENUM`
6. **Real-World Data** (Tasks 15–16): `LIKE`, `COALESCE`, arithmetic in SELECT
7. **Automation** (Tasks 17–18): `CREATE TRIGGER`, `BEFORE`/`AFTER`, `OLD`/`NEW` references

## Schema Datasets

| File | Contents |
|------|----------|
| `hbtn_0d_tvshows.sql` | TV shows + genres (many-to-many via junction table) |
| `hbtn_0d_tvshows_rate.sql` | Extended with ratings fact table |
| `metal_bands.sql` | Metal bands dataset with origin, fans, lifespan |
| `temperatures.sql` | Weather time-series by city, state, year, month |

## Resources

- [MySQL Documentation](https://dev.mysql.com/doc/)
