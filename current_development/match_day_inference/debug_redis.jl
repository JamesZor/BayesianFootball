using Pkg; Pkg.activate(".")
using Redis
using JSON3

redis_host = get(ENV, "REDIS_HOST", "127.0.0.1")
redis_port = parse(Int, get(ENV, "REDIS_PORT", "6379"))
redis_conn = RedisConnection(host=redis_host, port=redis_port)

meta_dict = Redis.hgetall(redis_conn, "live_market_meta")
println("Found $(length(meta_dict)) keys in live_market_meta")

for (k, v) in meta_dict
    if occursin("derry-city", v) && occursin("MATCH_ODDS", v)
        println("\n=== DERRY CITY MATCH_ODDS ===")
        println("Meta Key (market_id): ", k)
        println("Meta Value: ", v)
        
        live_data = Redis.hget(redis_conn, "live_markets", k)
        if isnothing(live_data)
            println("Live Data for $k: NOTHING (Not found in 'live_markets')")
        else
            data_str = String(live_data)
            println("Live Data Length: ", length(data_str))
            println("Live Data Preview: ", data_str[1:min(300, length(data_str))])
        end
    end
end
