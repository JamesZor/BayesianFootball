using Revise
using BayesianFootball

using DataFrames
using ThreadPinning
pinthreads(:cores)



# Load from sql
# server is busy... hence blacked out
# ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())

# working local
db_config = Data.DBConfig("postgresql://admin:supersecretpassword@192.168.1.88:5432/sofascrape_db")
db_conn =   Data.connect_to_db(db_config.url)
segment =   Data.ScottishLower()

try
    data_store = Data.get_datastore(db_conn, segment)
finally
    # Always close the connection, even if an error occurs during fetching
    close(db_conn) 
end



