# Test-only harness commands must verify libpq's resolved database, not just a parsed URL.
function assert_klm_test_database!(storage)
    storage.dbname == "mcmc_experiments_test" || error(
        "--test-db refuses non-test database $(storage.dbname)")
    conn = BayesianFootball.Training.Inference._db_connect(storage)
    try
        rows = BayesianFootball.Training.Inference._db_rows(conn,
            "SELECT current_database()::text AS dbname;")
        actual = String(only(rows.dbname))
        actual == "mcmc_experiments_test" || error(
            "--test-db connected to $actual, not mcmc_experiments_test; refusing any writes")
    finally
        close(conn)
    end
    return storage
end
