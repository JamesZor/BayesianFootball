#!/usr/bin/env python3
"""Small, deterministic CSV EDA. No DB connections or production fits.
Inputs: committed read-only Julia exports and existing inverted close CSVs.
Team reconstructions are constrained least squares, not new forecasting fits.
"""
from pathlib import Path
import csv, json, hashlib, math
from collections import defaultdict, Counter
from datetime import date
import numpy as np
import duckdb
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=HERE/'results'; FIG=HERE/'figures'
RNG=np.random.default_rng(20261011)
B=4000
SEASONS=['21/22','22/23','23/24','24/25','25/26']
MODELS=['td_lower_base','td_lower_a2full_carry_jump','grw_base','grw_step_a2_carry_jump','m12_td_ref','qs_marketobs']


def read(name):
    with open(OUT/name,newline='') as f:return list(csv.DictReader(f))


def write(name,rows,fields=None):
    rows=list(rows)
    if fields is None:
        fields=list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT/name,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def dt(s):return date.fromisoformat(str(s)[:10])
def prev(s):return f'{int(s[:2])-1:02d}/{int(s[3:])-1:02d}'
def season_date(s):
    d=dt(s);y=d.year-(d.month<7)
    return f'{y%100:02d}/{(y+1)%100:02d}'
def mean(x):return float(np.mean(x)) if len(x) else float('nan')
def sd(x):return float(np.std(x,ddof=1)) if len(x)>1 else float('nan')
def bins(g):return '1' if g==1 else '2-5' if g<=5 else '6-10' if g<=10 else '11-20' if g<=20 else '21+'


def bootstrap(rows,value='delta',cluster=('club','season')):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in cluster)].append(float(r[value]))
    vals=list(groups.values())
    if not vals:return (float('nan'),)*3
    sums=np.array([sum(x) for x in vals]);ns=np.array([len(x) for x in vals])
    ix=RNG.integers(len(vals),size=(B,len(vals)))
    boot=sums[ix].sum(1)/ns[ix].sum(1)
    return mean([v for x in vals for v in x]),*map(float,np.quantile(boot,[.025,.975]))


def summarize(rows,keys,value='delta',scale=1,cluster=('club','season')):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    result=[]
    for key,rr in sorted(groups.items()):
        m,lo,hi=bootstrap(rr,value,cluster)
        byclub=defaultdict(list)
        for r in rr:byclub[(r.get('club',''),r.get('season',''))].append(float(r[value]))
        loo=[mean([float(r[value]) for r in rr if (r.get('club',''),r.get('season',''))!=k]) for k in byclub] if len(byclub)>1 else [float('nan')]
        result.append(dict(zip(keys,key),mean=m*scale,lo=lo*scale,hi=hi*scale,n_obs=len(rr),n_fixtures=len(set(r['match_id'] for r in rr)),n_club_seasons=len(byclub),loo_min=min(loo)*scale,loo_max=max(loo)*scale))
    return result


# Use SQL for file normalization, joins and inventory, numerical Python for covariance/figures.
con=duckdb.connect()
qpath=ROOT/'current_development/market_model/results/QS/rates_sco_*.csv'
con.execute(f"CREATE TABLE quotes AS SELECT * FROM read_csv('{qpath}',union_by_name=true) WHERE season <= '25/26' AND date < DATE '2026-07-01'")
con.execute(f"CREATE TABLE fixtures AS SELECT * FROM read_csv('{OUT/'fixtures.csv'}') WHERE match_date < DATE '2026-07-01'")
con.execute("CREATE TABLE app AS SELECT match_id,tournament_id,season,match_date,home_team club,away_team opponent,'home' side FROM fixtures WHERE tournament_id BETWEEN 54 AND 57 UNION ALL SELECT match_id,tournament_id,season,match_date,away_team,home_team,'away' FROM fixtures WHERE tournament_id BETWEEN 54 AND 57")
con.execute("CREATE TABLE appearances AS SELECT *,row_number() OVER (PARTITION BY club,season ORDER BY match_date,match_id) game FROM app")
con.execute(f"COPY (SELECT a.club,a.season,a.tournament_id,count(*) league_games,count(q.match_id) inversion_rows,count(*) FILTER (WHERE q.accepted) accepted,count(*) FILTER (WHERE q.accepted=false) rejected,min(a.game) FILTER (WHERE q.accepted) first_quote_game,max(q.date) FILTER (WHERE q.accepted) last_quote_date,min(q.date) FILTER (WHERE q.accepted) first_quote_date FROM appearances a LEFT JOIN quotes q USING(match_id) GROUP BY ALL ORDER BY 3,2,1) TO '{OUT/'q6_coverage.csv'}' (HEADER)")
con.execute(f"COPY (SELECT q.*,a.game game_home,b.game game_away FROM quotes q JOIN appearances a ON a.match_id=q.match_id AND a.side='home' JOIN appearances b ON b.match_id=q.match_id AND b.side='away' WHERE q.accepted ORDER BY date,match_id) TO '{OUT/'accepted_quotes.csv'}' (HEADER)")
fixtures={int(r['match_id']):r for r in read('fixtures.csv')}
apps=con.execute('SELECT match_id,club,season,tournament_id,game,match_date,side,opponent FROM appearances').fetchall()
game={(m,c):int(g) for m,c,s,t,g,d,side,op in apps}
tiers={(c,s):int(t) for m,c,s,t,g,d,side,op in apps}
clubapps=defaultdict(list)
for m,c,s,t,g,d,side,op in apps:clubapps[(c,s)].append((m,dt(d),g))
quotes=read('accepted_quotes.csv')
quote_by_id={int(q['match_id']):q for q in quotes}
quoteclub=defaultdict(list)
for q in quotes:
    for c in [q['home'],q['away']]:quoteclub[(c,q['season'])].append(q)


def group(club,season):
    old=tiers.get((club,prev(season)),0);new=tiers.get((club,season),0)
    if not old:return 'entered_spfl'
    if old==new:return 'nonmover'
    if old==55 and new==56:return 'relegated_into_L1'
    if old==57 and new==56:return 'promoted_into_L1'
    return f'{old}_to_{new}'


def broad(club,season):
    old=tiers.get((club,prev(season)),0);new=tiers.get((club,season),0)
    return 'entrant' if not old else 'relegated' if old<new else 'promoted' if old>new else 'nonmover'


# q gauge sum(q)=0 per division; s on common mu_ref=0 scale, not league-centred.
# Difference response has intercept gamma/2, sum has intercept mu+gamma/2.
recons={};reconrows=[];graphrows=[]


def reconstruct(qs,tier,season,window):
    clubs=sorted(set(c for q in qs for c in (q['home'],q['away'])))
    n=len(clubs); ix={c:i for i,c in enumerate(clubs)}
    if n<3:return None
    contrast=np.vstack([np.eye(n-1),-np.ones(n-1)])
    xd=np.zeros((len(qs),n));xs=xd.copy();yd=[];ys=[]
    for k,q in enumerate(qs):
        hd=contrast[ix[q['home']]];ad=contrast[ix[q['away']]]
        xd[k,:-1]=hd-ad;xs[k,:-1]=hd+ad;xd[k,-1]=xs[k,-1]=1
        lh,la=math.log(float(q['lambda_h'])),math.log(float(q['lambda_a']))
        yd.append((lh-la)/2);ys.append((lh+la)/2)
    yd=np.array(yd);ys=np.array(ys)
    rankd=np.linalg.matrix_rank(xd);ranks=np.linalg.matrix_rank(xs)
    graphrows.append(dict(tier=tier,season=season,window=window,n_quotes=len(qs),n_clubs=n,q_rank=rankd,s_rank=ranks,expected_rank=n,q_condition=np.linalg.cond(xd),s_condition=np.linalg.cond(xs),connected_identified=rankd==n and ranks==n))
    if rankd<n or ranks<n or len(qs)<=n:return None
    bd=np.linalg.lstsq(xd,yd,rcond=None)[0];bs=np.linalg.lstsq(xs,ys,rcond=None)[0]
    ed=yd-xd@bd;es=ys-xs@bs
    vd=(ed@ed)/(len(qs)-n)*np.linalg.inv(xd.T@xd)
    vs=(es@es)/(len(qs)-n)*np.linalg.inv(xs.T@xs)
    # s_abs = s_centred + (mu - fixed mu_ref)/2, mu_ref=0.
    tq=np.column_stack([contrast,np.zeros(n)])
    ts=np.column_stack([contrast,np.ones(n)/2])
    qv=tq@bd;sv=ts@bs-bd[-1]/2
    cov_ds=float(ed@es)/(len(qs)-n)*np.linalg.inv(xd.T@xd)@xd.T@xs@np.linalg.inv(xs.T@xs)
    gamma_style_cov=(ts@cov_ds.T)[:,-1]
    vq=tq@vd@tq.T
    vstyle=ts@vs@ts.T+vd[-1,-1]/4-.5*(gamma_style_cov[:,None]+gamma_style_cov[None,:])
    cov_qs=tq@cov_ds@ts.T-.5*(tq@vd)[:,-1][:,None]
    result=dict(clubs=clubs,q=qv,s=sv,vq=vq,vs=vstyle,cov_qs=cov_qs,gamma=2*bd[-1],mu=bs[-1]-bd[-1],dates=[dt(q['date']) for q in qs],ids=[int(q['match_id']) for q in qs],resid_q=sd(ed),resid_s=sd(es))
    for i,c in enumerate(clubs):
        reconrows.append(dict(club=c,season=season,tier=tier,window=window,n_quotes=len(qs),n_own_quotes=sum(c in (q['home'],q['away']) for q in qs),q=qv[i],s=sv[i],q_se=math.sqrt(vq[i,i]),s_se=math.sqrt(vstyle[i,i]),q_s_cov=cov_qs[i,i],gamma=result['gamma'],mu=result['mu'],first_date=min(result['dates']),last_date=max(result['dates'])))
    return result


for s in SEASONS:
    for t in range(54,58):
        qq=sorted([q for q in quotes if int(q['tournament'])==t and q['season']==s],key=lambda q:(q['date'],int(q['match_id'])))
        if len(qq)<25:continue
        for name,part in [('whole',qq),('late80',qq[-80:]),('early60',qq[:60]),('late60',qq[-60:]),('mid60',qq[-120:-60])]:
            recons[(s,t,name)]=reconstruct(part,t,s,name)
write('q3_reconstruction_graph.csv',graphrows)
write('market_team_reconstruction.csv',reconrows)


def anchor(club,season,window='late80'):
    t=tiers.get((club,prev(season)))
    r=recons.get((prev(season),t,window))
    if not r or club not in r['clubs']:return dict(q=0.,s=0.,vq=.16,vs=.04,n=0,missing=True)
    i=r['clubs'].index(club)
    return dict(q=float(r['q'][i]),s=float(r['s'][i]),vq=float(r['vq'][i,i]),vs=float(r['vs'][i,i]),n=len(quoteclub[(club,prev(season))]),missing=False)


def historical_gamma(s,t):
    vals=[r['gamma'] for (ss,tt,w),r in recons.items() if ss<s and tt==t and w=='whole' and r]
    return mean(vals) if vals else 0.


# Transition inventory remains explicit even where destination or origin quotes are absent.
transitions=[]
for (c,s),t in sorted(tiers.items()):
    if s not in SEASONS or broad(c,s)=='nonmover':continue
    oldq=quoteclub[(c,prev(s))];newq=quoteclub[(c,s)]
    transitions.append(dict(club=c,season=s,old_tier=tiers.get((c,prev(s)),0),new_tier=t,group=group(c,s),broad=broad(c,s),old_quotes=len(oldq),new_quotes=len(newq),first_quote_game=min([game[(int(q['match_id']),c)] for q in newq],default=''),old_last_date=max([q['date'] for q in oldq],default=''),both_sides=bool(oldq and newq),prior_available=not anchor(c,s)['missing']))
write('q6_transitions.csv',transitions)

# Q1/Q5/Q10: club selections, correct persisted fold assignments and exact training manifests.
raw=[];cutoffcheck=[];trainingclubs={}
for model in MODELS:
    ff={int(r['match_id']):r for r in read('folds_'+model+'.csv')}
    lat={int(r['match_id']):r for r in read('rates_'+model+'.csv')}
    historical_ids=defaultdict(set)
    for h in read('history_'+model+'.csv'):
        mid=int(h['match_id']);f=fixtures.get(mid)
        if f and dt(f['match_date'])<date(2000+int(h['target_season'][:2]),7,1):historical_ids[int(h['fold'])].add(mid)
    historical_counts={fold:Counter(c for mid in ids for c in [fixtures[mid]['home_team'],fixtures[mid]['away_team']]) for fold,ids in historical_ids.items()}
    for r in read('observations_'+model+'.csv'):
        mid=int(r['match_id'])
        if r['family']!='1X2' or r['selection'] not in ('home','away') or mid not in fixtures:continue
        f=fixtures[mid];s=f['season'];t=int(f['tournament_id'])
        if t not in (56,57) or s not in ['23/24','24/25','25/26']:continue
        side=r['selection'];c=f[side+'_team'];a=anchor(c,s)
        meta=ff.get(mid);assigned=lat.get(mid)
        valid=bool(meta and assigned and dt(meta['cutoff'])<dt(f['match_date']))
        cutoffcheck.append(dict(model=model,match_id=mid,club=c,valid=valid))
        age=(dt(f['match_date'])-dt(meta['cutoff'])).days if valid else -1
        nown=sum(dt(meta['cutoff'])<d<dt(f['match_date']) for _,d,_ in clubapps[(c,s)]) if valid else -1
        g=game[(mid,c)];pm=float(r['p_model']);pk=float(r['p_market']);y=float(r['y'])
        oldscope=historical_counts.get(int(meta['fold']),Counter())[c] if valid else 0
        oldloweravailable=sum(1 for m,cc,ss,tt,gg,d,sd_,opp in apps if cc==c and ss<s and tt in (56,57))
        oldall=sum(len(qq) for (cc,ss),qq in quoteclub.items() if cc==c and ss<s)
        llm=-(y*math.log(max(pm,1e-12))+(1-y)*math.log(max(1-pm,1e-12)))
        llk=-(y*math.log(max(pk,1e-12))+(1-y)*math.log(max(1-pk,1e-12)))
        raw.append(dict(model=model,match_id=mid,club=c,season=s,tier=t,side=side,group=group(c,s),broad=broad(c,s),game=g,game_bin=bins(g),fold=int(meta['fold']),cutoff=meta['cutoff'] if valid else '',age_days=age,age_bin='0-7' if age<=7 else '8-14' if age<=14 else '15+' if age>=0 else 'missing',own_since_cutoff=nown,p_model=pm,p_market=pk,y=y,delta=pm-pk,absolute_delta=abs(pm-pk),market_residual=y-pk,model_residual=y-pm,ll_delta=llm-llk,brier_delta=(pm-y)**2-(pk-y)**2,old_in_scope=oldscope,old_lower_roster_available=oldloweravailable,old_alltier_quotes=oldall,history_class='no_fitted_history' if oldscope==0 else 'has_fitted_history',old_previous_quotes=a['n']))
write('q1_club_observations.csv',raw)
write('q12_prediction_chronology.csv',cutoffcheck)
assert all(r['valid'] for r in cutoffcheck),'prediction cutoff manifest invalid'
first20=[r for r in raw if r['game']<=20]
write('q1_bias_bins.csv',summarize(first20,['model','group','game_bin'],scale=100))
write('q1_bias_first20.csv',summarize(first20,['model','group'],scale=100))
write('q1_bias_by_game.csv',summarize(first20,['model','group','game'],scale=100))
write('q1_club_bias.csv',summarize(first20,['model','group','club','season'],scale=100))
write('q1_history_bias.csv',summarize(first20,['model','broad','history_class'],scale=100))
common=set(r['match_id'] for r in raw if r['model']=='qs_marketobs')
write('q1_common_panel.csv',summarize([r for r in first20 if r['match_id'] in common],['model','group'],scale=100))
pairs=[]
lookup={(r['model'],r['match_id'],r['club']):r for r in first20}
for base,arm in [('td_lower_base','td_lower_a2full_carry_jump'),('grw_base','grw_step_a2_carry_jump')]:
    for r in first20:
        if r['model']!=base:continue
        a=lookup.get((arm,r['match_id'],r['club']))
        if a:pairs.append(dict(r,model=arm+' minus '+base,delta=a['delta']-r['delta']))
write('q1_paired_a2.csv',summarize(pairs,['model','group','game_bin'],scale=100))
write('q5_age_bias.csv',summarize(first20,['model','group','game_bin','age_bin'],scale=100))
write('q5_age_absolute.csv',summarize(first20,['model','broad','age_bin'],value='absolute_delta',scale=100))
write('q5_own_since_cutoff.csv',summarize(first20,['model','broad','game_bin','own_since_cutoff'],scale=100))
write('q5_age_aggregate.csv',summarize(first20,['model','group','age_bin'],scale=100))
# Repeated prediction vintages for same fixture are not in these artifacts.
write('q5_vintage_availability.csv',[dict(model=m,n_unique_fixtures=len({r['match_id'] for r in raw if r['model']==m}),successive_vintages=0,interpretation='one held-out vintage per fixture; no causal refit event pairing') for m in MODELS])

# Q10 outcomes: two-mover fixtures not independent for proper scores; cluster by fixture.
for value in ['market_residual','model_residual','ll_delta','brier_delta']:
    write('q10_'+value+'.csv',summarize(first20,['model','group'],value=value,scale=100 if 'residual' in value else 1,cluster=('match_id',)))
write('q10_by_season.csv',summarize(first20,['model','group','season'],value='market_residual',scale=100))
calrows=[]
for r in first20:
    calrows.append(dict(r,price_bin=f'{min(4,int(r["p_market"]*5))*20}-{min(4,int(r["p_market"]*5))*20+20}'))
write('q10_calibration.csv',summarize(calrows,['model','broad','side','price_bin'],value='market_residual',scale=100))
# Price/home/season matching with non-movers, descriptive stratification, fixture bootstrap.
matchrows=[]
for model in MODELS:
    rr=[r for r in calrows if r['model']==model]
    controls=defaultdict(list)
    for r in rr:
        if r['broad']=='nonmover':controls[(r['season'],r['side'],r['price_bin'])].append(r)
    for r in rr:
        if r['broad']=='nonmover':continue
        cs=controls[(r['season'],r['side'],r['price_bin'])]
        if cs:matchrows.append(dict(r,delta=r['market_residual']-mean([c['market_residual'] for c in cs]),n_controls=len(cs)))
write('q10_matched_market_residual.csv',summarize(matchrows,['model','group'],scale=100))

# Full 1X2 categorical proper score (one outcome per fixture), not selected-club binary loss.
proper=[]
for model in MODELS:
    tri=defaultdict(dict)
    for r in read('observations_'+model+'.csv'):
        if r['family']=='1X2':tri[int(r['match_id'])][r['selection']]=r
    for mid,rr in tri.items():
        if set(rr)!=set(['home','draw','away']) or mid not in fixtures:continue
        f=fixtures[mid]
        if int(f['tournament_id']) not in (56,57):continue
        movers=[c for c in [f['home_team'],f['away_team']] if broad(c,f['season'])!='nonmover' and game[(mid,c)]<=20]
        winner=next((r for r in rr.values() if float(r['y'])==1),None)
        if winner:
            proper.append(dict(model=model,match_id=mid,club='fixture',season=f['season'],subset='mover_first20' if movers else 'other',delta=math.log(float(winner['p_market']))-math.log(float(winner['p_model'])),n_movers=len(movers)))
write('q10_fixture_logloss.csv',summarize(proper,['model','subset'],cluster=('match_id',)))

# Q4: equal-count market reconstruction windows with analytic opponent covariance.
summer=[]
for (c,s),t in sorted(tiers.items()):
    if s not in SEASONS or broad(c,s)!='nonmover':continue
    for typ,a_key,b_key in [('summer',(prev(s),t,'late60'),(s,t,'early60')),('within',(prev(s),t,'mid60'),(prev(s),t,'late60'))]:
        a=recons.get(a_key);b=recons.get(b_key)
        if not a or not b or c not in a['clubs'] or c not in b['clubs']:continue
        ia=a['clubs'].index(c);ib=b['clubs'].index(c)
        overlap=len(set(a['ids'])&set(b['ids']))
        # Overlapping same-season estimates have correlated errors: do not subtract independent noise.
        # Keep them as an explicitly unadjusted sensitivity, not evidence of process noise.
        summer.append(dict(club=c,season=s,tier=t,type=typ,dq=b['q'][ib]-a['q'][ia],ds=b['s'][ib]-a['s'][ia],vq=a['vq'][ia,ia]+b['vq'][ib,ib],vs=a['vs'][ia,ia]+b['vs'][ib,ib],gap_days=(min(b['dates'])-max(a['dates'])).days,midpoint_gap_days=(sum(d.toordinal() for d in b['dates'])/len(b['dates'])-sum(d.toordinal() for d in a['dates'])/len(a['dates'])),overlap_quotes=overlap,first_n=len(a['ids']),second_n=len(b['ids']),first_date=max(a['dates']),second_date=min(b['dates'])))
write('q4_window_changes.csv',summer)
q4=[]
for typ in ['summer','within']:
    for tier in ['all',54,55,56,57]:
        rr=[r for r in summer if r['type']==typ and (tier=='all' or r['tier']==tier) and r['overlap_quotes']==0]
        for coord,var in [('dq','vq'),('ds','vs')]:
            v=[float(r[coord]) for r in rr];sampling=mean([r[var] for r in rr]);observed=sd(v)**2
            q4.append(dict(type=typ,tier=tier,coordinate=coord,n_club_seasons=len(rr),mean=mean(v),observed_sd=sd(v),mean_estimation_variance=sampling,noise_adjusted_sd=math.sqrt(max(0,observed-sampling)),mean_gap_days=mean([r['gap_days'] for r in rr]),mean_midpoint_gap=mean([r['midpoint_gap_days'] for r in rr]),limitations='shared-opponent covariance retained in window estimates; league-normalisation uncertainty still approximate; independent seasons/windows only'))
write('q4_dispersion.csv',q4)

# Posterior boundary step from existing QS r_k/tau_k; square posterior mean proxy explicitly labelled.
scales=read('q4_qs_scales.csv');byfold=defaultdict(dict)
for r in scales:byfold[int(r['fold'])][r['name']]=float(r['mean'])
qss=[]
for f,rr in byfold.items():
    tau=rr.get('dyn.τₖ');ratio=rr.get('dyn.rₖ')
    if tau is None or ratio is None:continue
    qss.append(dict(fold=f,tau_k=tau,r_k=ratio,q_step_sd=tau/math.sqrt(1+ratio**2),s_step_sd=tau*ratio/math.sqrt(1+ratio**2),method='posterior-mean plug-in one ordinary biweek; not full posterior predictive covariance'))
write('q4_qs_boundary_proxy.csv',qss)

# Q2: genuinely earlier-season-only, direct-fixture D/U predictions. Common s reference mu=0.
# Non-mover opponents use previous endpoint; missing opponents use prior-season division mean style.
GROUPS=['relegated','promoted','entrant']

def fallback_style(s,t):
    vals=[r['mu']/2 for (ss,tt,w),r in recons.items() if ss<s and tt==t and w=='whole' and r]
    return mean(vals) if vals else 0.


def old_state(c,s):
    a=anchor(c,s)
    if a['missing']:a['s']=fallback_style(s,tiers[(c,s)])
    return a


def feature(q,coordinate,carry):
    s=q['season'];h=q['home'];a=q['away'];ah=old_state(h,s);aa=old_state(a,s)
    gh=broad(h,s);ga=broad(a,s)
    v='q' if coordinate=='D' else 's'
    sign=-1 if coordinate=='D' else 1
    base=(ah[v] if gh=='nonmover' else carry*ah[v]) + sign*(aa[v] if ga=='nonmover' else carry*aa[v])
    feat=np.array([float(gh==g)+sign*float(ga==g) for g in GROUPS])
    old= (ah[v] if gh!='nonmover' else 0)+sign*(aa[v] if ga!='nonmover' else 0)
    variance=ah['vq' if v=='q' else 'vs']+aa['vq' if v=='q' else 'vs']
    return base,feat,old,variance


def target(q,coordinate):
    lh=math.log(float(q['lambda_h']));la=math.log(float(q['lambda_a']));gam=historical_gamma(q['season'],int(q['tournament']))
    return (lh-la-gam)/2 if coordinate=='D' else (lh+la-gam)/2


opening=[]
for q in quotes:
    s=q['season']
    if s=='21/22':continue
    moving=[c for c in [q['home'],q['away']] if broad(c,s)!='nonmover' and game[(int(q['match_id']),c)]<=5]
    if moving:opening.append(q)


def learn(s,coord,method):
    training=[q for q in opening if q['season']<s]
    carry=0 if method=='P1' else 1
    # One fixed weak ridge (equivalent 2 opening observations) stabilises sparse direction cells.
    # No panel-tuned tuning constants. Carry sensitivity separately reported.
    x=[];y=[]
    for q in training:
        base,f,old,var=feature(q,coord,carry)
        if method=='P2_carry_est':f=np.r_[f,old]
        x.append(f);y.append(target(q,coord)-base)
    nparam=4 if method=='P2_carry_est' else 3
    center=np.array([.235,-.235,0.]) if method=='P2_A2shrink' and coord=='D' else np.zeros(3)
    if not x:return dict(offset=center,carry=carry,sigma=.35 if coord=='D' else .2,cov=np.eye(3)*(.35 if coord=='D' else .2)**2,n=0,seasons=[],status='unlearned A2-centered fallback' if method=='P2_A2shrink' else 'unlearned broad fallback')
    x=np.array(x);y=np.array(y)
    ridge=np.eye(nparam)*2
    beta=np.linalg.solve(x.T@x+ridge,x.T@y+(ridge@center if nparam==3 else np.zeros(nparam)))
    if method=='P2_carry_est':
        newcarry=float(np.clip(1+beta[-1],0,1))
        # Re-estimate offsets conditional on clipped carry.
        x3=x[:,:3];y2=[]
        for q in training:
            base,_,_,_=feature(q,coord,newcarry);y2.append(target(q,coord)-base)
        beta3=np.linalg.solve(x3.T@x3+np.eye(3)*2,x3.T@np.array(y2))
        residual=np.array(y2)-x3@beta3;carry=newcarry;x=x3;beta=beta3;ridge=np.eye(3)*2
    else:residual=y-x@beta
    # Leave quote noise/opponent uncertainty in predictive residual, never call it intrinsic jump.
    sigma=max(.08 if coord=='D' else .05,math.sqrt(float(residual@residual)/max(1,len(residual)-3)))
    cov=sigma**2*np.linalg.inv(x.T@x+ridge)
    return dict(offset=beta,carry=carry,sigma=sigma,cov=cov,n=len(training),seasons=sorted(set(q['season'] for q in training)),status='earlier seasons only')


methods=['league_mean','no_shift','A2_step','P1','P2','P2_A2shrink','P2_carry_est','P3_fixed_step_residual','results_tier']
params=[];priorrows=[];manifest=[]
# A2 comparison is STEP-ONLY: no selected W2 posterior carry/jump constants.
for s in SEASONS[1:]:
    for method in ['P1','P2','P2_A2shrink','P2_carry_est']:
        for coord in ['D','U']:
            p=learn(s,coord,method)
            for i,g in enumerate(GROUPS):params.append(dict(target_season=s,method=method,coordinate=coord,group=g,carry=p['carry'],offset=p['offset'][i],sigma_predictive=p['sigma'],offset_se=math.sqrt(p['cov'][i,i]),n_training_fixtures=p['n'],training_seasons=','.join(p['seasons']),status=p['status']))
    for q in [q for q in quotes if q['season']==s]:
        mid=int(q['match_id']);movers=[c for c in [q['home'],q['away']] if broad(c,s)!='nonmover']
        if not movers:continue
        for c in movers:
            # first accepted quote, then quote nearest actual games <=5/10/20 (no pretending late quotes are openers).
            qq=sorted(quoteclub[(c,s)],key=lambda x:game[(int(x['match_id']),c)])
            chosen={1:int(qq[0]['match_id'])}
            for k in [5,10,20]:
                qqk=[x for x in qq if game[(int(x['match_id']),c)]<=k]
                if qqk:chosen[k]=int(qqk[-1]['match_id'])
            checkpoints=[k for k,m in chosen.items() if m==mid]
            if not checkpoints:continue
            for coord in ['D','U']:
                actual=target(q,coord)
                for method in methods:
                    base,f,old,var=feature(q,coord,1)
                    p=learn(s,coord,'P2' if method not in ['P1','P2_carry_est','P2_A2shrink'] else method)
                    if method in ['P1','P2','P2_carry_est','P2_A2shrink']:
                        base,f,old,var=feature(q,coord,p['carry']);pred=base+f@p['offset'];sig=math.sqrt(p['sigma']**2+f@p['cov']@f)
                    else:
                        pred=base;sig=math.sqrt(p['sigma']**2+var)
                        sign=-1 if coord=='D' else 1
                        for club,sgn in [(q['home'],1),(q['away'],sign)]:
                            g=broad(club,s);oldtier=tiers.get((club,prev(s)),0);newtier=tiers[(club,s)];a=old_state(club,s)
                            if g=='nonmover':continue
                            if method=='league_mean':pred-=sgn*a['q' if coord=='D' else 's'];pred+=sgn*(0 if coord=='D' else fallback_style(s,newtier))
                            if method in ['A2_step','results_tier'] and oldtier:
                                pred+=sgn*(.235*(newtier-oldtier) if coord=='D' else -.0094*(newtier-oldtier))
                            if method=='results_tier':
                                # Old league results net-goal rank proxy, separately labelled.
                                matches=[fixtures[m] for m,_,_ in clubapps.get((club,prev(s)),[]) if m in fixtures]
                                gd=mean([(int(r['home_score'])-int(r['away_score']))*(1 if r['home_team']==club else -1) for r in matches]) if matches else 0
                                # First-order log-rate quality at prior-season goals environment; predeclared conversion.
                                oldmu=recons.get((prev(s),oldtier,'whole'));mu=oldmu['mu'] if oldmu else 0
                                resultq=gd/(2*math.exp(mu))
                                if coord=='D':pred+=sgn*(resultq-a['q'])
                        if method=='P3_fixed_step_residual':
                            # With free residual direction jumps, a fixed tier rebase is algebraically absorbed;
                            # this is the P2-identical sensitivity, NOT an identified P3 estimate.
                            pred=base+f@p['offset'];sig=math.sqrt(p['sigma']**2+f@p['cov']@f)
                    err=pred-actual
                    for checkpoint in checkpoints:
                        priorrows.append(dict(club=c,season=s,match_id=mid,opponent=q['away'] if c==q['home'] else q['home'],side='home' if c==q['home'] else 'away',group=group(c,s),broad=broad(c,s),checkpoint=checkpoint,actual_game=game[(mid,c)],coordinate=coord,method=method,prediction=pred,actual=actual,error=err,abs_error=abs(err),squared_error=err**2,predictive_sd=sig,covered95=abs(err)<=1.96*sig,log_score=-norm.logpdf(actual,loc=pred,scale=sig),old_quotes=anchor(c,s)['n'],opponent_old_quotes=anchor(q['away'] if c==q['home'] else q['home'],s)['n'],n_training_fixtures=p['n'],training_seasons=','.join(p['seasons'])))
            # Source IDs/dates for both old club states plus all older hyperparameter labels.
            stateids=set()
            for club in [q['home'],q['away']]:
                rr=recons.get((prev(s),tiers.get((club,prev(s))),'late80'))
                if rr:stateids.update(rr['ids'])
            labelids={int(x['match_id']) for x in opening if x['season']<s}
            # Historical gamma and fallback division-style estimates also use earlier quotes.
            normalizationids={int(x['match_id']) for x in quotes if x['season']<s}
            allids=stateids|labelids|normalizationids
            manifest.append(dict(target_season=s,club=c,match_id=mid,source_ids=';'.join(map(str,sorted(allids))),state_source_ids=';'.join(map(str,sorted(stateids))),hyper_label_ids=';'.join(map(str,sorted(labelids))),normalization_source_ids=';'.join(map(str,sorted(normalizationids))),max_source_date=max([quote_by_id[m]['date'] for m in allids],default=''),source_seasons=','.join(sorted({quote_by_id[m]['season'] for m in allids})),target_node_date=f'20{s[:2]}-07-01',violations=sum(quote_by_id[m]['season']>=s or dt(quote_by_id[m]['date'])>=date(2000+int(s[:2]),7,1) for m in allids),fallback=anchor(c,s)['missing'],input_hash=hashlib.sha256(json.dumps(sorted(allids)).encode()).hexdigest(),gauge='q sum-zero by old division; s mu_ref=0; destination relative (pooled mapping unresolved)'))
write('q2_prior_predictions.csv',priorrows)
write('q2_prior_parameters.csv',params)
write('q12_source_manifest.csv',manifest)
assert all(r['violations']==0 for r in manifest),'prior chronology violation'


def prior_summary(rows,keys):
    gg=defaultdict(list)
    for r in rows:gg[tuple(r[k] for k in keys)].append(r)
    result=[]
    for key,rr in sorted(gg.items()):
        err=np.array([r['error'] for r in rr]);pp=np.array([r['prediction'] for r in rr]);yy=np.array([r['actual'] for r in rr]);clusters={(r['club'],r['season']) for r in rr}
        slope=np.polyfit(pp,yy,1)[0] if len(rr)>=3 and np.std(pp)>1e-8 else float('nan')
        result.append(dict(zip(keys,key),n_obs=len(rr),n_fixtures=len({r['match_id'] for r in rr}),n_club_seasons=len(clusters),bias=mean(err),mae=mean(abs(err)),rmse=math.sqrt(mean(err**2)),coverage95=mean([r['covered95'] for r in rr]),mean_predictive_sd=mean([r['predictive_sd'] for r in rr]),negative_log_score=mean([r['log_score'] for r in rr]),calibration_slope=slope,old_history_missing=sum(r['old_quotes']==0 for r in rr),opponent_history_missing=sum(r['opponent_old_quotes']==0 for r in rr)))
    return result

write('q2_prior_summary.csv',prior_summary(priorrows,['method','coordinate','checkpoint']))
write('q2_prior_season.csv',prior_summary(priorrows,['method','coordinate','checkpoint','season']))
write('q2_prior_direction.csv',prior_summary(priorrows,['method','coordinate','checkpoint','broad']))
write('q2_coverage_sensitivity.csv',prior_summary([r for r in priorrows if r['old_quotes']>=10 and r['opponent_old_quotes']>=10],['method','coordinate','checkpoint']))
write('q8_carry_forward.csv',prior_summary([r for r in priorrows if r['method'] in ['P1','P2','P2_carry_est'] and r['checkpoint'] in [1,5]],['method','coordinate','checkpoint','season']))
# First-five quote MEAN direct D/U loss, not merely the nearest quote at game 5.
first5=[]
for q in opening:
    s=q['season'];mid=int(q['match_id'])
    for c in [q['home'],q['away']]:
        if broad(c,s)=='nonmover' or game[(mid,c)]>5:continue
        for coord in ['D','U']:
            y=target(q,coord)
            for method in ['league_mean','A2_step','no_shift','P1','P2','P2_A2shrink','P2_carry_est']:
                p=learn(s,coord,method if method in ['P1','P2','P2_A2shrink','P2_carry_est'] else 'P2')
                base,f,old,var=feature(q,coord,p['carry'] if method in ['P1','P2_carry_est'] else 1)
                pred=base+(f@p['offset'] if method in ['P1','P2','P2_A2shrink','P2_carry_est'] else 0)
                sign=-1 if coord=='D' else 1
                for club,sgn in [(q['home'],1),(q['away'],sign)]:
                    if broad(club,s)=='nonmover':continue
                    a=old_state(club,s);ot=tiers.get((club,prev(s)),0);nt=tiers[(club,s)]
                    if method=='league_mean':pred+=sgn*((0 if coord=='D' else fallback_style(s,nt))-a['q' if coord=='D' else 's'])
                    if method=='A2_step' and ot:pred+=sgn*(.235*(nt-ot) if coord=='D' else -.0094*(nt-ot))
                sig=math.sqrt(p['sigma']**2+f@p['cov']@f)
                first5.append(dict(club=c,season=s,match_id=mid,coordinate=coord,method=method,checkpoint='mean_first5',actual_game=game[(mid,c)],prediction=pred,actual=y,error=pred-y,covered95=abs(pred-y)<=1.96*sig,predictive_sd=sig,log_score=-norm.logpdf(y,pred,sig),old_quotes=anchor(c,s)['n'],opponent_old_quotes=anchor(q['away'] if c==q['home'] else q['home'],s)['n']))
write('q2_first5_predictions.csv',first5)
write('q2_first5_summary.csv',prior_summary(first5,['method','coordinate','checkpoint']))

# Q7 mover re-rating: descriptive old/new seasonal relative q/s plus chronological anchored opening shifts.
shifts=[]
for t in transitions:
    c=t['club'];s=t['season'];old=recons.get((prev(s),t['old_tier'],'late80'));new=recons.get((s,t['new_tier'],'whole'))
    if not old or not new or c not in old['clubs'] or c not in new['clubs']:continue
    i=old['clubs'].index(c);j=new['clubs'].index(c)
    dq=float(new['q'][j]-old['q'][i]);ds=float(new['s'][j]-old['s'][i])
    shifts.append(dict(t,dq=dq,ds=ds,dalpha=dq+ds,dbeta=ds-dq,vq=old['vq'][i,i]+new['vq'][j,j],vs=old['vs'][i,i]+new['vs'][j,j],q_s_measurement_cov=old['cov_qs'][i,i]+new['cov_qs'][j,j],label='descriptive whole-destination-season; never used as opening prior training label'))
write('q7_mover_shifts.csv',shifts)
q7=[]
for g in sorted(set(r['broad'] for r in shifts)):
    rr=[r for r in shifts if r['broad']==g];dq=[r['dq'] for r in rr];ds=[r['ds'] for r in rr]
    vq=max(0,sd(dq)**2-mean([r['vq'] for r in rr]));vs=max(0,sd(ds)**2-mean([r['vs'] for r in rr]))
    q7.append(dict(broad=g,n_club_seasons=len(rr),mean_dq=mean(dq),mean_ds=mean(ds),q_sd=sd(dq),s_sd=sd(ds),noise_adjusted_q_sd=math.sqrt(vq),noise_adjusted_s_sd=math.sqrt(vs),style_quality_sd_ratio=math.sqrt(vs/vq) if vq else float('nan'),dq_ds_cov=float(np.cov(dq,ds)[0,1]) if len(rr)>1 else float('nan'),noise_adjusted_dq_ds_cov=(float(np.cov(dq,ds)[0,1])-mean([r['q_s_measurement_cov'] for r in rr])) if len(rr)>1 else float('nan'),mean_attack_share=mean([r['dalpha']/(2*r['dq']) for r in rr if abs(r['dq'])>.05]),limitations='division-relative q shift confounds tier rebase and club jump; plug-in uncertainty, not independent club samples'))
write('q7_shift_summary.csv',q7)
# Opening direct quote minus frozen old fixture prediction, oriented for moving club.
open_shift=[]
for r in priorrows:
    if r['method']!='no_shift':continue
    sign=1 if r['side']=='home' or r['coordinate']=='U' else -1
    open_shift.append(dict(r,delta=-sign*r['error']))
write('q7_opening_shift.csv',summarize(open_shift,['broad','coordinate','checkpoint']))

# Q8 measurement reliability sensitivity and selection overlap, not a causal claim.
q8=[]
for coord in ['q','s']:
    for g in ['relegated','promoted']:
        rr=[r for r in shifts if r['broad']==g]
        xx=[];yy=[];vv=[]
        for r in rr:
            a=anchor(r['club'],r['season']);xx.append(a[coord]);yy.append(a[coord]+r['d'+coord]);vv.append(a['v'+coord])
        if len(xx)<3:continue
        naive=float(np.cov(xx,yy,ddof=1)[0,1]/np.var(xx,ddof=1));rel=1-mean(vv)/np.var(xx,ddof=1)
        q8.append(dict(broad=g,coordinate=coord,n=len(xx),slope_naive=naive,old_measurement_reliability=rel,slope_error_corrected=naive/rel if rel>0 else float('nan'),old_min=min(xx),old_max=max(xx),interpretation='whole-destination descriptive slope only; forward carry choice in q8_carry_forward'))
write('q8_selection_slopes.csv',q8)

# Q12 overlap and season-freeze manifest. Current QS already uses many old rows.
overlap=[]
hist=read('history_qs_marketobs.csv');foldhist=defaultdict(set);foldseason={}
likelihood_ids={int(r['match_id']) for r in read('q12_qs_market_likelihood.csv') if r['full_book']=='true'}
for r in hist:foldhist[int(r['fold'])].add(int(r['match_id']));foldseason[int(r['fold'])]=r['target_season']
for fold,ids in sorted(foldhist.items()):
    s=foldseason[fold];mm=[r for r in manifest if r['target_season']==s]
    sources=set(int(x) for r in mm for x in r['source_ids'].split(';') if x)
    oldstate=set(int(x) for r in mm for x in r['state_source_ids'].split(';') if x)
    overlap.append(dict(fold=fold,target_season=s,n_prior_source_quotes=len(sources),n_state_source_quotes=len(oldstate),n_overlap_history=len(sources&ids&likelihood_ids),n_state_overlap_history=len(oldstate&ids&likelihood_ids),n_overlap_training_fixtures=len(sources&ids),chronology_violations=sum(r['violations'] for r in mm),prior_inputs_identical_across_folds=True,policy='BLOCK naive historical-init prior; use disjoint pre-window data or sequential boundary with no re-counted quote rows; EB overlap must be labelled separately',historical_initial_node_earliest=min([fixtures[m]['match_date'] for m in ids if m in fixtures],default=''),proposed_boundary_node=f'20{s[:2]}-07-01'))
write('q12_overlap.csv',overlap)
write('q12_status.csv',[dict(check='source season and source date before target boundary',violations=sum(r['violations'] for r in manifest),status='pass'),dict(check='prediction fixture after actual latest training row',violations=sum(not r['valid'] for r in cutoffcheck),status='pass'),dict(check='prior fixed by target season, not refit',violations=0,status='pass by construction'),dict(check='naive prior plus existing historical market likelihood',violations=sum(r['n_overlap_history']>0 for r in overlap),status='BLOCK; implementation dependency'),dict(check='future-informed historical window initial node',violations=sum(r['historical_initial_node_earliest']<r['proposed_boundary_node'] for r in overlap),status='BLOCK placement at historical initial node'),dict(check='cup quote archive timestamp available',violations=1,status='unresolved; cups descriptive only'),dict(check='26/27 outcome/performance diagnostics',violations=0,status='not performed; calendar-year cup labels removed by date embargo')])

# Optional questions are records, never invented results.
write('q11_squad_status.csv',[dict(status='not run (optional)',reason='prioritised prerequisites and Q1-Q8/Q10; retained-minutes lineage not exported; no claim about squad mechanism')])

# Figures generated solely from CSV-backed tables.
for model in ['qs_marketobs','td_lower_base','td_lower_a2full_carry_jump']:
    plt.figure(figsize=(7,4))
    for g in ['relegated_into_L1','promoted_into_L1','nonmover']:
        rr=[r for r in read('q1_bias_by_game.csv') if r['model']==model and r['group']==g]
        if rr:plt.plot([int(r['game']) for r in rr],[float(r['mean']) for r in rr],marker='.',label=g)
    plt.axhline(0,color='grey',lw=.7);plt.xlabel('Actual league game');plt.ylabel('Model minus market win probability (pp)');plt.legend(fontsize=8);plt.title(model);plt.tight_layout();plt.savefig(FIG/f'q1_{model}.png',dpi=150);plt.close()
plt.figure(figsize=(7,4))
rr=read('q2_prior_summary.csv')
for coord in ['D','U']:
    part=[r for r in rr if r['coordinate']==coord and r['checkpoint']=='1' and r['method'] in ['league_mean','A2_step','P1','P2','P2_A2shrink','P2_carry_est']]
    plt.plot([r['method'] for r in part],[float(r['rmse']) for r in part],marker='o',label=coord)
plt.ylabel('First accepted quote RMSE (log-rate rotation units)');plt.xticks(rotation=15);plt.legend();plt.tight_layout();plt.savefig(FIG/'q2_prior_rmse.png',dpi=150);plt.close()
plt.figure(figsize=(6,4))
for g in ['relegated','promoted']:
    rr=[r for r in shifts if r['broad']==g]
    plt.scatter([r['dq'] for r in rr],[r['ds'] for r in rr],label=g)
plt.axhline(0,color='grey');plt.axvline(0,color='grey');plt.xlabel('Destination-relative q minus old endpoint q');plt.ylabel('Common-reference s change');plt.legend();plt.tight_layout();plt.savefig(FIG/'q7_mover_shifts.png',dpi=150);plt.close()
print('ANALYSIS_OK quotes=',len(quotes),'bias_rows=',len(raw),'prior_predictions=',len(priorrows),'manifest=',len(manifest))
