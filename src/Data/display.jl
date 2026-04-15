# src/dataV2/display.jl


# ==============================================================================
# PRETTY PRINTING (Data Containers)
# ==============================================================================

function Base.show(io::IO, ::MIME"text/plain", ds::DataStore)
    # Title & Segment
    printstyled(io, "DataStore", color=:cyan, bold=true)
    printstyled(io, " [$(typeof(ds.segment))]", color=:yellow, bold=true)
    printstyled(io, "\n (Container for football data)\n", color=:light_black)
    println(io, "=========")

    # Define a helper to print each dataframe section consistently
    function print_df_summary(label, df)
        # Section Header with Dimensions
        printstyled(io, "[$label] ", color=:magenta)
        if isempty(df)
            printstyled(io, "(Empty)\n", color=:light_black)
        else
            dims = "$(nrow(df)) rows × $(ncol(df)) cols"
            printstyled(io, "($dims)\n", color=:white)
            
            # Column Preview (First 5-6 columns to avoid clutter)
            cols = names(df)
            preview_len = min(6, length(cols))
            preview_cols = join(string.(Symbol.(cols[1:preview_len])), ", ")
            remaining = length(cols) - preview_len
            
            print(io, "  Columns: ")
            printstyled(io, preview_cols, color=:light_black)
            if remaining > 0
                printstyled(io, " + $remaining more", color=:light_black, italic=true)
            end
            println(io)
        end
    end

    # Print the five main sections
    print_df_summary("Matches", ds.matches)
    println(io)
    print_df_summary("Statistics", ds.statistics)
    println(io)
    print_df_summary("Odds", ds.odds)
    println(io)
    print_df_summary("Lineups", ds.lineups)
    println(io)
    print_df_summary("Incidents", ds.incidents)
end

# Compact inline show (for arrays/logging)
function Base.show(io::IO, ds::DataStore)
    seg = typeof(ds.segment)
    m = nrow(ds.matches)
    s = nrow(ds.statistics)
    o = nrow(ds.odds)
    l = nrow(ds.lineups)
    i = nrow(ds.incidents)
    
    print(io, "DataStore[$seg](matches=$m, stats=$s, odds=$o, lineups=$l, incidents=$i)")
end



