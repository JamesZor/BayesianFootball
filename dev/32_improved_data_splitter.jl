# dev/32_improved_data_splitter.jl

"""

We define season data as Sⁱ where i defines the years/ season index so that
S²⁰ imples that we have the seaosn "20/21" . 

futher more, we add to this notation with S^{i}_{n:m} or S^{i}_{n} with n,m postive intergers 
that denote the match weeks ( or the time dynamics of the season ) with n being the start and m being the end 
of the indexing so: 
s^{20} = { S^{20}_{1} , S^{20}_{2} , S^{20}_{3} , ..., S^{20}_{end} } = S^{20}_{1:end} 
with S^{20}_{1} being a collection of features of the games played in that week. 
So just looking at the 20/21 season. 

Hence we want to be able to construct a set of sets in order to conduct expanding window cross validations 
/ walk forward test on a season, to simulate the data avabiles in real time series manner, with the n 
indicating the warm up number of time dynamics to be considers before the first predictions p_1 which is 
index at the n+1. Then it follows that m is the last training batch need either for end -1 so we can predict 
p_end as we the last week of the season is not need since the there are no game after that - only the next season, or 
at the end in the case we want to make a predict at m+1. 
Abusing some set notation we have 

S^{20}_{n:end} = { { S^{20}_{1} }, 
                  { S^{20}_{1} + S^{20}_{2} },
                  { S^{20}_{1} + S^{20}_{2} + S^{20}_{3} }, 
 ..., 
                  { S^{20}_{1} + S^{20}_{2} + ... +  S^{20}_{end} }
                }
noting that the operator "+" is more of a contacation of the data sets. 

S^{20:21}_{n:end} = { { S^{20}_{1} }, 
                  { S^{20}_{1} + S^{20}_{2} },
                  { S^{20}_{1} + S^{20}_{2} + S^{20}_{3} }, 
 ..., 
                  { S^{20}_{1} + S^{20}_{2} + ... +  S^{20}_{end} }

                  { S^{21}_{1} }, 
                  { S^{21}_{1} + S^{21}_{2} },
                  { S^{21}_{1} + S^{21}_{2} + S^{21}_{3} }, 
 ..., 
                  { S^{21}_{1} + S^{21}_{2} + ... +  S^{21}_{end} }
                }

In the data split we want to indicate if the training / data split includes historical years, 
as in if we include the previous season in the model, i.e we split 20/21 seaosn based on the time 
dynamics but include the 19/20 seaosns for the model, we can denotes this t which is T-t for the seaosn 
to include,  S^{21:t:21}_{n:end} or  S^{20:t:21}_{n:end} 

normal : 
S^{20:0:20}_{n:end} = { { S^{20}_{1} }, 
                  { S^{20}_{1} + S^{20}_{2} },
                  { S^{20}_{1} + S^{20}_{2} + S^{20}_{3} }, 
 ..., 
                  { S^{20}_{1} + S^{20}_{2} + ... +  S^{20}_{end} } 
}

the case to include one year previous t=1 

S^{21:1:21}_{n:end} = { { S^{20} + S^{21}_{1} }, 
                  { S^{20} + S^{21}_{1} + S^{21}_{2} },
                  { S^{21}_{1} + S^{21}_{2} + S^{21}_{3} }, 
 ..., 
                  { S^{20} + S^{21}_{1} + S^{21}_{2} + ... +  S^{21}_{end} } 
}

the case for t=1 but repeat for the seaosn 20/21 and 21/22: 


S^{20:1:21}_{n:end} = 
                  { { S^{19} + S^{20}_{1} }, 
                  { S^{19} + S^{20}_{1} + S^{20}_{2} },
                  { S^{20}_{1} + S^{20}_{2} + S^{20}_{3} }, 
 ..., 
                  { S^{19} + S^{20}_{1} + S^{20}_{2} + ... +  S^{20}_{end} },

                  { { S^{20} + S^{21}_{1} }, 
                  { S^{20} + S^{21}_{1} + S^{21}_{2} },
                  { S^{21}_{1} + S^{21}_{2} + S^{21}_{3} }, 
 ..., 
                  { S^{20} + S^{21}_{1} + S^{21}_{2} + ... +  S^{21}_{end} } 
}


The aim is to allow use to create this kind of data splits for the training process in our models. 

Following the data.create_data_splits api function structure 
we have a mapping from create_data_splits: D × C -> D' 
were D is the data_store, and C is a config , D' is like a powerset of the D, / or a view with will be like the S. 

here for the data_split config 
data_split_config :
  tournament_id = vector /list  -> [1] for just tournament_id =1 or [1],[2] to repeat the process for tournament_id 
                1 and 2 seperately, or [1,2] to do 1,2 at the same time. 

  dynamics_col = symbol to indicate which dataframe column to use for the time dynamics so moslty either month or week 

  warnup_period_dynamics = n -> where to begin as doing an ar1 process on 1 week wont fit well so we allow the seaosn to play be for training 

  seasons => list of seasons to be split up examples: ["20/21"], ["20/21", "21/22"]  etc repeats for seaosn and tournament_id 

  end_daynamics: m -> if we need to run to the need to make a predcit m+1 or stop a week before the last week, as no games after the last week, end or end -1 

  season_hist: i -> the number of past seaosn to include, examples: 0 imples that it will be just "20/21" or the seaosns indicated 
                    in the seaons parameter, if 1 then use the last season so ["20/21"] we need to include ["19/20"] S^{20:1:20}_{n:end}, 
                    noting we need to check/ ensure we dont go back to far as currelty we have the data for 20/21 til 24/25 amd some of 25/26. 
                    for the time being so need ot check in data store and warn users that it cant be done if i is too large. 


D' is similar to the out now, a vector of tuples with index 1 is the S^{20}_{n} ( the training data split) 
and the [2] is currelty a string, however this string should be a split_meta_data struct, which inherts from abstract meta struct 
as we need one for data split and then the feature_split - tho they will be the same i think. 
the point of the split_meta_data is to contain relavent information regarding the split for the features split process which 
which will mostly be the split_meta_data information, and which is need in the predictions part of the process 
so we can ensure we are predicting the correct season, and time step 
so we need to carry this information 
split_meta_data: 
    -tournament_ids 
    -current_seaosn_fold:
    -current_time_step:


in the mapping create_data_splits, here it would be nice to filter / subset the need data from the data_store 
struct, as it will help reduce boiler code when writitng experients. 



"""
