# Definitions only; no fitting and no latent regeneration.
module DecompressionEvaluation

import ..DecompressionPXG
import DataFrames as DF
import Statistics as S

const D = DecompressionPXG
const CAL = D.BayesianFootball.Calibration

function market_reference(odds, panel)
    return CAL.inversion_frame(CAL.invert_market_rates(odds; match_ids = panel))
end

"""
Regress market supremacy on model supremacy with an intercept. The reported
`market_on_model_slope` is the work package's decompression diagnostic: 1.0 is
scale agreement and values above 1.0 mean the model is compressed. The reverse
slope is retained so the result is directly comparable with diagnostics that use
model-on-market orientation.
"""
function decompression(fit, odds, ds, rates)
    latents = fit.latents
    model_frame = DF.DataFrame(
        match_id = latents.match_ids,
        model_supremacy = vec(S.mean(log.(latents.λ_home) .- log.(latents.λ_away); dims = 2)),
    )
    accepted = DF.filter(:accepted => identity, rates)
    joined = DF.innerjoin(model_frame, accepted; on = :match_id)
    DF.nrow(joined) >= 10 || error("fewer than 10 accepted market inversions")
    model = joined.model_supremacy
    market = log.(joined.lambda_mkt_h) .- log.(joined.lambda_mkt_a)
    model_centered = model .- S.mean(model)
    market_centered = market .- S.mean(market)
    model_ss = sum(abs2, model_centered)
    market_ss = sum(abs2, market_centered)
    model_ss > 0.0 && market_ss > 0.0 || error("zero supremacy variance")
    cross = sum(model_centered .* market_centered)
    market_on_model_slope = cross / model_ss
    model_on_market_slope = cross / market_ss
    market_on_model_intercept = S.mean(market) - market_on_model_slope * S.mean(model)
    r2 = cross^2 / (model_ss * market_ss)
    through_origin_slope = sum(model .* market) / sum(abs2, model)

    ppd = D.BayesianFootball.Predictions.model_inference(
        latents, fit.config.model; market_config = D.PXG_MARKETS)
    probabilities = DF.DataFrame(
        match_id = Int.(ppd.df.match_id),
        selection = Symbol.(ppd.df.selection),
        market_name = String.(ppd.df.market_name),
        p_model = S.mean.(ppd.df.distribution),
    )
    wins = DF.filter(
        row -> row.market_name == "1X2" && row.selection in (:home, :away), probabilities)
    closing = DF.filter(
        row -> row.market_name == "1X2" && row.selection in (:home, :away), odds)
    favourites = DF.innerjoin(
        wins,
        DF.select(closing, :match_id, :selection, :prob_fair_close);
        on = [:match_id, :selection],
    )
    DF.filter!(:prob_fair_close => probability -> probability >= 0.70, favourites)
    outcomes = Dict(
        Int(row.match_id) => (Int(row.home_score), Int(row.away_score))
        for row in DF.eachrow(ds.matches)
    )
    favourites.realized = [begin
        home, away = outcomes[Int(row.match_id)]
        row.selection == :home ? Float64(home > away) : Float64(away > home)
    end for row in DF.eachrow(favourites)]
    n_favourites = DF.nrow(favourites)
    summary = (;
        market_on_model_slope,
        market_on_model_intercept,
        through_origin_slope,
        model_on_market_slope,
        r2,
        n_inverted = DF.nrow(joined),
        n_inversion_refused = count(!, rates.accepted),
        n_favourites,
        favourite_model = n_favourites == 0 ? NaN : S.mean(favourites.p_model),
        favourite_market = n_favourites == 0 ? NaN : S.mean(favourites.prob_fair_close),
        favourite_realized = n_favourites == 0 ? NaN : S.mean(favourites.realized),
    )
    return summary, joined, favourites
end

function capital_allocation(result)
    bets = result.trajectory.bets
    total = sum(bets.stake)
    fraction(mask) = total > 0.0 ? sum(bets.stake[mask]) / total : NaN
    return (;
        capital_ge4 = fraction(bets.odds .>= 4.0),
        capital_le18 = fraction(bets.odds .<= 1.8),
        stake_fraction_sum = total,
    )
end

end # module
