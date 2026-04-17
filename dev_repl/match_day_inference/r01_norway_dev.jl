
using JLD2

JLD2.save_object("norway_datastore_test.jld2", ds)

ds_backup = JLD2.load_object("norway_datastore_test.jld2")
