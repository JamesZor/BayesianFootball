#!/usr/bin/env python3
"""Draw-wise native-Poisson m12 pricing for the r08 transition panel.

The selected m12 model is declared PoissonCountModel with a joint Gamma-Poisson
observation: its score grid is therefore independent Poisson.  This script
reproduces the repository's 12x12 score support (goals 0:11), draw by draw,
then averages 1X2 prices.  It does not substitute a score grid at mean lambda.
"""
from __future__ import annotations
import csv, math, struct, os, json
from collections import defaultdict
from pathlib import Path
try:
    import zstandard
except ImportError:
    raise SystemExit("Install zstandard (python -m pip install --user zstandard); no pricing performed.")
import psycopg

SUITE = Path("experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda")
DATA, RESULTS = SUITE / "data", SUITE / "results"
RUN_ID = "928dad3b-ccaf-4909-b6b7-4f1a815e1cab"
PORTFOLIO_ID = "a7c4c55b-f8d2-416e-ba85-9c7fe9bedc1a"
COMMISSION, FRACTION, BANKROLL0 = 0.02, 0.25, 100.0


def read(path: Path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))
def write(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def mean(v): return f"{sum(v)/len(v):.4f}" if v else ""
def decode(blob: bytes):
    # Julia CodecZstd frames need not advertise content size; streaming decode
    # handles both that form and ordinary size-bearing frames.
    import io
    raw=zstandard.ZstdDecompressor().stream_reader(io.BytesIO(blob)).read()
    if len(raw) < 10 or raw[:4] != b"BFCL" or raw[4] != 1: raise ValueError("unsupported CountLatents draw blob")
    if raw[5] != 0: raise ValueError("observation parameters present: Poisson-only decoder refuses this blob")
    n = struct.unpack_from("<I",raw,6)[0]
    if not n or len(raw) != 10+16*n: raise ValueError("invalid draw count or trailing bytes")
    off=10; home=struct.unpack_from(f"<{n}d",raw,off); off += 8*n
    away=struct.unpack_from(f"<{n}d",raw,off)
    if not all(math.isfinite(x) and x >= 0 for x in home+away): raise ValueError("invalid Poisson rate")
    return home, away

def one_x_two(home, away):
    # Exact support used by compute_score_matrix(...; max_goals=12), without
    # renormalizing omitted >=12-goal tail mass.
    ph=pd=pa=0.0
    for lh, la in zip(home, away):
        h=[math.exp(-lh)]; a=[math.exp(-la)]
        for k in range(1,12): h.append(h[-1]*lh/k); a.append(a[-1]*la/k)
        for i in range(12):
            for j in range(12):
                q=h[i]*a[j]
                if i>j: ph += q
                elif i==j: pd += q
                else: pa += q
    n=len(home); return ph/n,pd/n,pa/n

def main():
    transitions=read(RESULTS / "r08_transition_all_fixture_panel.csv")
    # One team-side row per match window; source candidate changes cannot duplicate.
    # Preserve BOTH transitioning club sides in the same fixture.
    wanted=sorted({int(r["match_id"]) for r in transitions})
    prices={r["match_id"]:r for r in read(DATA / "r01_betfair_1x2_last_coherent_preko.csv")}
    with psycopg.connect(os.getenv("BF_EXPERIMENTS_DB_URL", "host=mcmc-beast port=5432 dbname=mcmc_experiments user=postgres"), connect_timeout=10,
                         options="-c default_transaction_read_only=on -c statement_timeout=30000 -c lock_timeout=5000") as c:
        config=c.execute("SELECT model_config FROM configs WHERE config_id=%s",(RUN_ID,)).fetchone()
        if not config or not config[0].get('type','').startswith('PoissonCountModel{') or config[0].get('fields',{}).get('observation') != 'JointGammaPoissonObservation()':
            raise ValueError("saved model family does not match audited Poisson route")
        latent=c.execute("""SELECT ml.match_id,ml.draws_blob FROM match_latents ml
          JOIN fold_results fr ON fr.fold_id=ml.fold_id WHERE fr.run_id=%s AND ml.match_id=ANY(%s)""",(RUN_ID,wanted)).fetchall()
        bets=c.execute("""SELECT pb.match_id,pb.kickoff_date,pb.market_family,pb.selection,pb.odds_close,pb.stake_amount,pb.pnl
          FROM portfolio_bets pb JOIN portfolio_runs pr USING(portfolio_run_id)
          WHERE pr.model_run_id=%s AND pr.portfolio_run_id=%s""",(RUN_ID,PORTFOLIO_ID)).fetchall()
    if len(latent) != len({int(mid) for mid,_ in latent}): raise ValueError("duplicate held-out match latent")
    latent={int(mid):blob for mid,blob in latent}
    grids={mid:one_x_two(*decode(blob)) for mid,blob in latent.items()}
    outputs=[]
    for r in transitions:
        mid=int(r['match_id'])
        if mid not in latent: continue
        ph,pd,pa=grids[mid]
        is_home=r["is_home"] == "1"; pteam=ph if is_home else pa
        price=prices.get(str(mid)); odds=float(price["odds_home"] if is_home else price["odds_away"]) if price else None
        pmarket=float(price["fair_p_home"] if is_home else price["fair_p_away"]) if price else None
        b=(odds-1)*(1-COMMISSION) if odds else None
        kelly=max(0.0,(pteam*(b+1)-1)/b) if b else None
        outputs.append({**r,"m12_draw_averaged_team_win_probability":f"{pteam:.6f}","m12_score_support":"0:11 independent Poisson draw-wise",
                        "score_support_mass":f"{ph+pd+pa:.10f}",
                        "m12_opponent_win_probability":f"{pa if is_home else ph:.6f}",
                        "coherent_market_opponent_probability":"" if not price else price['fair_p_away' if is_home else 'fair_p_home'],
                        "opponent_archive_odds":"" if not price else price['odds_away' if is_home else 'odds_home'],
                        "transitioning_market_favourite":"" if not price else str(int(pmarket > max(float(price['fair_p_draw']),float(price['fair_p_away' if is_home else 'fair_p_home'])))),
                        "coherent_market_team_probability":"" if pmarket is None else f"{pmarket:.6f}","m12_minus_market":"" if pmarket is None else f"{pteam-pmarket:.6f}",
                        "archive_odds":"" if odds is None else f"{odds:.4f}","quarter_kelly_fraction":"" if kelly is None else f"{FRACTION*kelly:.6f}"})
    fields=list(outputs[0]) if outputs else ["match_id"]
    write(RESULTS / "r09_m12_transition_draw_pricing.csv",outputs,fields)
    summary=[]
    for (direction,window),g in sorted(group(outputs,lambda r:(r["movement_direction"],r["window"])).items()):
        mkt=[float(r["coherent_market_team_probability"]) for r in g if r["coherent_market_team_probability"]]
        delta=[float(r["m12_minus_market"]) for r in g if r["m12_minus_market"]]
        summary.append({"movement_direction":direction,"window":window,"transition_rows":len(g),"m12_latent_coverage_of_transition_panel":f"{len(g)/len([x for x in transitions if x['movement_direction']==direction and x['window']==window]):.3f}","coherent_market_coverage":f"{len(mkt)/len(g):.3f}","paired_rows":len(mkt),"mean_m12_team_win_probability":mean([float(r['m12_draw_averaged_team_win_probability']) for r in g if r['coherent_market_team_probability']]),"mean_market_team_probability":mean(mkt),"mean_m12_minus_market":mean(delta)})
    write(RESULTS / "r09_m12_transition_market_summary.csv",summary,list(summary[0]))
    # Existing portfolio strategy subset: only actual 1X2 bets on the transitioning side.
    lookup={(int(r['match_id']), 'home' if r['is_home']=='1' else 'away'):r for r in outputs}
    actual=[]
    for mid,date,family,selection,odds,stake,pnl in bets:
        r=lookup.get((int(mid),selection))
        if r and family.startswith("1X2"):
            actual.append({"portfolio_run_id":PORTFOLIO_ID,"analysis_side":"transitioning_team","club":r['club'],"match_id":mid,"kickoff_date":date,"movement_direction":r["movement_direction"],"window":r["window"],"selection":selection,"odds_close":odds,"stake_amount":stake,"pnl":pnl})
    write(RESULTS / "r09_existing_portfolio_transition_team_bets.csv",actual,list(actual[0]) if actual else ["match_id"])
    # Clearly counterfactual, sequential single-team 1X2 quarter Kelly at archive
    # snapshot odds, not the correlated portfolio allocator and not T-25 execution.
    # Audit correction: use independent 100-unit exposure per row, NOT a sequential
    # bankroll: two club sides may share one fixture and are dependent exposures.
    simulated=[]
    for r in sorted(outputs,key=lambda x:(x['match_date'],int(x['match_id']))):
        if not r['archive_odds']: continue
        f=float(r['quarter_kelly_fraction']); stake=BANKROLL0*f; won=int(float(r['goal_difference'])>0)
        pnl=stake*((float(r['archive_odds'])-1)*(1-COMMISSION) if won else -1.0)
        simulated.append({"club":r['club'],"analysis_side":"transitioning_team","match_id":r['match_id'],"match_date":r['match_date'],"movement_direction":r['movement_direction'],"window":r['window'],"stake_fraction":f"{f:.6f}","stake":f"{stake:.4f}","pnl":f"{pnl:.4f}","counterfactual":"independent 100-unit capital per clubside; archive snapshot; 2% commission; no portfolio return"})
    write(RESULTS / "r09_simplified_quarter_kelly_transition.csv",simulated,list(simulated[0]) if simulated else ["match_id"])
    # Opponent-only diagnostic: back the underdog AGAINST a transitioning favourite.
    # Favourite is strict highest de-vigged archive 1X2 probability (ties excluded).
    opponents=[]; opponent_actual=[]
    for r in outputs:
        if r['transitioning_market_favourite'] != '1': continue
        p=float(r['m12_opponent_win_probability']); market=float(r['coherent_market_opponent_probability'])
        odds=float(r['opponent_archive_odds']); b=(odds-1)*(1-COMMISSION)
        fraction=FRACTION*max(0.,(p*(b+1)-1)/b); stake=BANKROLL0*fraction
        pnl=stake*(b if float(r['goal_difference'])<0 else -1.)
        selection='away' if r['is_home']=='1' else 'home'
        base={"club":r['club'],"opponent":r['opponent'],"match_id":r['match_id'],"movement_direction":r['movement_direction'],"window":r['window'],"analysis_side":"opponent_underdog_against_transitioning_archive_favourite","selection":selection}
        opponents.append({**base,"m12_probability":p,"market_probability":market,"m12_minus_market":p-market,"odds":odds,"quarter_kelly_fraction":fraction,"independent_100_unit_stake":stake,"independent_pnl":pnl})
        for mid,date,family,sel,close,actual_stake,actual_pnl in bets:
            if int(mid)==int(r['match_id']) and sel==selection and family.startswith('1X2'):
                opponent_actual.append({**base,"portfolio_run_id":PORTFOLIO_ID,"kickoff_date":date,"odds_close":close,"stake_amount":actual_stake,"pnl":actual_pnl})
    write(RESULTS / 'r09_m12_opponent_underdog_pricing.csv',opponents,list(opponents[0]) if opponents else ['match_id'])
    write(RESULTS / 'r09_m12_existing_portfolio_opponent_bets.csv',opponent_actual,list(opponent_actual[0]) if opponent_actual else ['match_id'])
    opponent_summary=[]
    for (direction,window),g in sorted(group(opponents,lambda r:(r['movement_direction'],r['window'])).items()):
        a=[r for r in opponent_actual if r['movement_direction']==direction and r['window']==window]
        opponent_summary.append({'movement_direction':direction,'window':window,'analysis_side':'opponent_underdog_against_transitioning_archive_favourite','paired_rows':len(g),'mean_m12_probability':mean([r['m12_probability'] for r in g]),'mean_market_probability':mean([r['market_probability'] for r in g]),'mean_m12_minus_market':mean([r['m12_minus_market'] for r in g]),'positive_kelly_rows':sum(r['quarter_kelly_fraction']>0 for r in g),'independent_stake_sum':sum(r['independent_100_unit_stake'] for r in g),'independent_pnl_sum':sum(r['independent_pnl'] for r in g),'existing_portfolio_bets':len(a),'existing_stake_sum':sum(float(r['stake_amount']) for r in a),'existing_pnl_sum':sum(float(r['pnl']) for r in a)})
    write(RESULTS / 'r09_m12_opponent_underdog_summary.csv',opponent_summary,list(opponent_summary[0]) if opponent_summary else ['paired_rows'])
    audit={'run_id':RUN_ID,'portfolio_run_id':PORTFOLIO_ID,'saved_model_config':config[0],
           'transition_clubside_rows':len(transitions),'priced_clubside_rows':len(outputs),'priced_unique_matches':len(grids),
           'paired_clubside_rows':sum(bool(r['archive_odds']) for r in outputs),'portfolio_total_bets':len(bets),
           'opponent_favourite_rows':len(opponents),'opponent_existing_bets':len(opponent_actual),
           'minimum_score_support_mass':min(sum(p) for p in grids.values()),
           'kernel_source':'src/predictions/score_grids/kernels.jl:53-102; types.jl:2; legacy score_computation/poisson.jl:29-62',
           'kernel_contract':'goals 0:11; independent Poisson per draw; no renormalization; recurrence algebraic parity, not Julia bit parity',
           'limitations':['archive snapshot differs from saved portfolio TWA[-20,0]','no causal model-lag inference','clubside rows dependent; isolated stakes not portfolio return','saved model config + BFCL route audited; no Julia serialized fit loaded']}
    (RESULTS / 'r09_m12_pricing_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('saved_model_config','limitations')},indent=2))

def group(rows,key):
    d=defaultdict(list)
    for r in rows:d[key(r)].append(r)
    return d
if __name__=="__main__":
    try: main()
    except Exception as exc:
        raise SystemExit(f"Pricing failed ({type(exc).__name__}); database/error details suppressed.") from None
