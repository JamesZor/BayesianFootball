# Definitions only; no fitting and no latent regeneration.
module MomentumEvaluation
import ..MomentumGRW
import DataFrames as DF
import Statistics as S
const M = MomentumGRW
const CAL = M.BayesianFootball.Calibration

function market_reference(odds,panel)
    rates = CAL.inversion_frame(CAL.invert_market_rates(odds;match_ids=panel))
    return rates
end

function decompression(fit,odds,ds,rates)
    lat = fit.latents
    sup = DF.DataFrame(match_id=lat.match_ids,
        model_supremacy=vec(S.mean(log.(lat.λ_home).-log.(lat.λ_away);dims=2)))
    accepted = DF.filter(:accepted=>identity,rates)
    joined = DF.innerjoin(sup,accepted;on=:match_id)
    DF.nrow(joined)>=10 || error("fewer than 10 accepted market inversions")
    x = log.(joined.lambda_mkt_h).-log.(joined.lambda_mkt_a)
    y = joined.model_supremacy
    xc = x.-S.mean(x)
    yc = y.-S.mean(y)
    xx = sum(abs2,xc)
    yy = sum(abs2,yc)
    xx>0 && yy>0 || error("zero supremacy variance")
    xy = sum(xc.*yc)
    slope = xy/xx

    ppd = M.BayesianFootball.Predictions.model_inference(lat,fit.config.model;market_config=M.MARKETS)
    probs = DF.DataFrame(match_id=Int.(ppd.df.match_id),selection=Symbol.(ppd.df.selection),
        market_name=String.(ppd.df.market_name),p_model=S.mean.(ppd.df.distribution))
    wins = DF.filter(r->r.market_name=="1X2" && r.selection in (:home,:away),probs)
    closing = DF.filter(r->r.market_name=="1X2" && r.selection in (:home,:away),odds)
    favourites = DF.innerjoin(wins,DF.select(closing,:match_id,:selection,:prob_fair_close);on=[:match_id,:selection])
    DF.filter!(:prob_fair_close=>p->p>=0.70,favourites)
    # Tail coverage does not depend on successful rate inversion.
    outcomes = Dict(Int(r.match_id)=>(Int(r.home_score),Int(r.away_score)) for r in DF.eachrow(ds.matches))
    favourites.realized = [begin
        h,a = outcomes[Int(r.match_id)]
        r.selection==:home ? Float64(h>a) : Float64(a>h)
    end for r in DF.eachrow(favourites)]
    n = DF.nrow(favourites)
    summary = (;slope,intercept=S.mean(y)-slope*S.mean(x),r2=xy^2/(xx*yy),
        n_inverted=DF.nrow(joined),n_inversion_refused=count(!,rates.accepted),
        n_favourites=n,favourite_model=n==0 ? NaN : S.mean(favourites.p_model),
        favourite_market=n==0 ? NaN : S.mean(favourites.prob_fair_close),
        favourite_realized=n==0 ? NaN : S.mean(favourites.realized))
    return summary,joined,favourites
end

function capital_allocation(result)
    bets = result.trajectory.bets
    total = sum(bets.stake)
    fraction(mask) = total>0 ? sum(bets.stake[mask])/total : NaN
    return (;capital_ge4=fraction(bets.odds.>=4.0),
        capital_le18=fraction(bets.odds.<=1.8),stake_fraction_sum=total)
end
end
