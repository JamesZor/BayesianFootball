#!/usr/bin/env python3
"""Lightweight offline kernel/decoder/artifact regression checks; no database/MCMC."""
import math
import struct
import unittest
from collections import Counter
import r09_m12_transition_pricing as p


class PricingAuditTests(unittest.TestCase):
    def test_decoder(self):
        raw=b'BFCL'+bytes([1,0])+struct.pack('<I4d',2,1.,2.,3.,4.)
        pack=p.zstandard.ZstdCompressor().compress
        self.assertEqual(p.decode(pack(raw)),((1.,2.),(3.,4.)))
        for bad in [b'',raw+b'x',raw[:5]+b'\x01'+raw[6:],raw[:5]+b'\x02'+raw[6:]]:
            with self.assertRaises(ValueError): p.decode(pack(bad))
        for rate in [-1.,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):
                p.decode(pack(b'BFCL'+bytes([1,0])+struct.pack('<I2d',1,rate,1.)))

    def test_kernel_against_independent_log_pmf(self):
        # Same source support/partition, independent log-gamma PMF evaluation.
        hs=[.01,.5,1.,2.,5.,10.]; aws=[8.,3.,1.,.1,4.,12.]
        sums=[[],[],[]]
        for h,a in zip(hs,aws):
            for i in range(12):
                for j in range(12):
                    q=math.exp(-h+i*math.log(h)-math.lgamma(i+1)-a+j*math.log(a)-math.lgamma(j+1))
                    sums[0 if i>j else 1 if i==j else 2].append(q)
        expected=[math.fsum(x)/len(hs) for x in sums]
        for x,y in zip(p.one_x_two(hs,aws),expected): self.assertAlmostEqual(x,y,places=12)
        self.assertEqual(p.one_x_two([0.],[0.]),(0.,1.,0.))
        self.assertLess(sum(p.one_x_two([12.],[12.])),.5)  # no normalization
        mix=p.one_x_two([.1,5.],[3.,.2]); at_mean=p.one_x_two([2.55],[1.6])
        self.assertGreater(max(abs(x-y) for x,y in zip(mix,at_mean)),.01)

    def test_artifact_cohorts(self):
        panel=p.read(p.RESULTS/'r08_transition_all_fixture_panel.csv')
        rows=p.read(p.RESULTS/'r09_m12_transition_draw_pricing.csv')
        mids={r['match_id'] for r in rows}
        key=lambda r:(r['match_id'],r['club'],r['window'])
        self.assertEqual(Counter(map(key,rows)),Counter(key(r) for r in panel if r['match_id'] in mids))
        self.assertEqual(len(rows),220)
        self.assertEqual(len(mids),196)
        for s in p.read(p.RESULTS/'r09_m12_transition_market_summary.csv'):
            paired=[r for r in rows if r['movement_direction']==s['movement_direction'] and r['window']==s['window'] and r['archive_odds']]
            self.assertEqual(len(paired),int(s['paired_rows']))
            for field,source in [('mean_m12_team_win_probability','m12_draw_averaged_team_win_probability'),('mean_market_team_probability','coherent_market_team_probability')]:
                self.assertEqual(s[field],p.mean([float(r[source]) for r in paired]))
        opp=p.read(p.RESULTS/'r09_m12_opponent_underdog_pricing.csv')
        self.assertEqual(len(opp),91)
        lookup={key(r):r for r in rows}
        for r in opp:
            source=lookup[key(r)]
            self.assertEqual(source['transitioning_market_favourite'],'1')
            self.assertEqual(r['selection'],'away' if source['is_home']=='1' else 'home')
            self.assertLess(float(r['market_probability']),float(source['coherent_market_team_probability']))
        actual=p.read(p.RESULTS/'r09_m12_existing_portfolio_opponent_bets.csv')
        self.assertEqual(len(actual),65)
        self.assertTrue(all(r['portfolio_run_id']==p.PORTFOLIO_ID for r in actual))


if __name__=='__main__': unittest.main()
