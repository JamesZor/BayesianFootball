# models_julia


# 🗄️ Data Module (SQL Datastore Pipeline)

This module handles the extraction, transformation, and validation of raw PostgreSQL data into memory-optimized Julia `DataFrames`. It serves as the foundational data layer for the `BayesianFootball` package.

## Basic Usage

The primary entry point for the module is `load_datastore_sql`. It takes a specific tournament segment, executes 5 concurrent SQL queries (`@async`), processes the results, and returns a unified `DataStore` object.

```julia
using BayesianFootball

# 1. Fetch all data for the Scottish Lower leagues
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())

# 2. Access the individual DataFrames
matches_df = ds.matches
odds_df    = ds.odds
```

## 📦 The `DataStore` Object

The `DataStore` is a strictly typed container that holds the data for the requested segment. Every DataFrame inside the `DataStore` has been heavily optimized using `InlineStrings` and strict schemas to minimize RAM usage.

* `segment::DataTournemantSegment` - The specific slice of data requested (e.g., `ScottishLower()`).
* `matches::DataFrame` - Core match details, scores, dates, and xG presence.
* `statistics::DataFrame` - Match-level statistics (possession, shots, etc.) pivoted into wide `_home` and `_away` format.
* `odds::DataFrame` - Betting market data, fully enriched by the `Markets` module (includes implied probabilities, vig removal, fair odds, and Closing Line Movement).
* `lineups::DataFrame` - Player-level starting XI, substitutes, and individual JSON performance stats.
* `incidents::DataFrame` - Event-level timeline data (goals, cards, substitutions, VAR decisions).

---

## ⚙️ Architecture & The Data Flow

To ensure data integrity, every single data domain (Matches, Odds, etc.) passes through a strict 3-step pipeline defined in `fetchers/interfaces.jl`. 

When you call `load_data(conn, segment, MatchesData())`, the orchestrator does the following:

1. **FETCH (`fetch_data`)**: Executes the raw SQL query via `LibPQ` to pull data for the specific `tournament_ids`.
2. **PROCESS (`process_data`)**: Performs the Julia-side ETL. This includes calculating dates, pivoting wide formats, applying mathematical enrichments (like the `Markets` probability math), and running `apply_schema!` to enforce strict column types.
3. **QA VALIDATE (`validate_data`)**: Acts as a "Data Contract." It checks the final DataFrame to ensure critical columns exist and that there are no illogical values (e.g., decimal odds below 1.0). If QA fails, it throws a loud warning.

All specific implementations for these 3 steps live in the `src/data/fetchers/sql/` folder.

---

## 🛠️ How to Add a New Segment (League)

Adding a new league or grouping of tournaments to the package is incredibly easy. You only need to touch **one file**.

**File:** `src/data/fetchers/segments.jl`

**Step 1: Define the Singleton Struct**
Create a new struct that subtypes `DataTournemantSegment`.
```julia
struct EnglishPremier <: DataTournemantSegment end 
```

**Step 2: Map the Tournament IDs**
Define the specific database IDs that belong to this segment by extending the `tournament_ids` function.
```julia
tournament_ids(::EnglishPremier) = [17]
```

That is it. You can immediately call `load_datastore_sql(EnglishPremier())`, and the entire pipeline will automatically target tournament ID 17.

---

## 📂 Directory Structure Reference

```text
src/data/
├── data-module.jl             # Top-level module: exports and includes
├── types.jl                   # Global structs (DataStore, DBConfig)
├── utils.jl                   # Generic helpers (apply_schema!)
├── Markets/                   # Probability, Vig, and CLM math engine
├── fetchers/
│   ├── schemas.jl             # Memory-optimized column type dictionaries
│   ├── segments.jl            # DataTournemantSegment definitions
│   ├── interfaces.jl          # The 3-step Fetch -> Process -> QA contract
│   ├── datastore.jl           # Manages DB connection and @async concurrency
│   └── sql/                   # The domain-specific SQL and ETL logic
│       ├── matches.jl         
│       ├── statistics.jl      
│       ├── lineups.jl         
│       ├── incidents.jl       
│       └── odds.jl            # Merges SQL data with the Markets module
```
