# Data Collection - APIs

Playing with paginated JSON APIs (`requests`, SWAPI, GitHub) to retrieve and
transform data — the first piece of a data lake.

## Tasks

| File | Prototype | Description |
|------|-----------|-------------|
| `0-passengers.py` | `def availableShips(passengerCount):` | Ships carrying at least `passengerCount` passengers (SWAPI starships, paginated) |
| `1-sentience.py` | `def sentientPlanets():` | Home-planet names of all `sentient` species (SWAPI species + planets, paginated) |
| `2-user_location.py` | script `arg = GitHub API user URL` | Print user `location`; `Not found` on 404; `Reset in X min` on 403 rate limit |

See `RESOURCES.md` for the ingested intranet resources.
