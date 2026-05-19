# current_development/calibration_module/data_l2_prep.jl


"""
Basic function to get the data ready for the calibration layer model. 
#TODO: - possible add the the pipeline for rolling features etc. 
"""
function build_l2_training_df(ds::Data.DataStore, ppd::Predictions.PPD)::AbstractDataFrame

   transform!(ppd.df,
      :distribution => ByRow(median) => :prob_median,
      :distribution => ByRow(mean) => :prob_mean )

    matches_df = select(ds.matches,
        :match_id, :match_date, :season, :match_month, 
        :home_score, :away_score, :home_team, :away_team,
        # process the split id 
        [:season, :match_month] => ByRow((s, m) -> string(s, "-", lpad(m, 2, "0"))) => :split_id;
        copycols=false # prevents copying the other columns

    )

    df = innerjoin(ppd.df, matches_df, on=[:match_id])

    return innerjoin(df, 
              select(ds.odds, 
                :match_id, :selection, :is_winner, 
                copycols=false),
              on=[:match_id, :selection])
end





