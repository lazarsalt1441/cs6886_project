import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from gpu_selection import select, calibrate

class SelectionTest(unittest.TestCase):
    def test_reject_regression(self):
        self.assertEqual(select({'base':100,'masked':300},'base'),'base')
    def test_noise_margin(self):
        self.assertEqual(select({'base':100,'opt':98},'base'),'base')
    def test_choose_gain(self):
        self.assertEqual(select({'base':100,'conditional':70,'masked':300},'base'),'conditional')
    def test_invalid_latency(self):
        for n in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):select({'base':n},'base')
    def test_missing_base(self):
        with self.assertRaises(ValueError):select({'opt':10},'base')
    def test_calibration_count(self):
        counts={'base':0,'opt':0}
        def fn(v):counts[v]+=1
        r=calibrate({v:lambda v=v:fn(v) for v in counts},'base',trials=7)
        self.assertEqual(counts,{'base':7,'opt':7})
        self.assertTrue(all(len(s)==7 for s in r['samples_us'].values()))

if __name__=='__main__':unittest.main()
