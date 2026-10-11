# Read-only structural metadata, never held-back outcomes.
s1audit("cup_metadata") do conn
    ENV["PGOPTIONS"]="-c default_transaction_read_only=on"
    c=LibPQ.Connection(ENV["BF_DB_URL"])
    try
        keys=QSBacktestPhase0.read_query(c,"SELECT DISTINCT jsonb_object_keys(raw_data::jsonb) AS json_key FROM sofascore.matches WHERE tournament_id IN (73,982,1520) AND start_timestamp >= '2021-07-01' AND start_timestamp < '2026-07-01' AND raw_data IS NOT NULL")
        CSV.write(joinpath(S1OUT,"q3_cup_json_keys.csv"),keys)
        println("CUP_JSON_KEYS=",join(keys.json_key,","))
    finally
        close(c)
    end
end
