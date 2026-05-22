using Pkg; Pkg.activate(".")
using Redis
using JSON3

redis_host = get(ENV, "REDIS_HOST", "127.0.0.1")
redis_port = parse(Int, get(ENV, "REDIS_PORT", "6379"))
redis_conn = RedisConnection(host=redis_host, port=redis_port)

keys = Redis.keys(redis_conn, "*")
println("All Redis Keys: ", keys)
