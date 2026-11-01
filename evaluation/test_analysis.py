import gc,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from analyze_saved import summarize
from run_followup import make_plan
class Tests(unittest.TestCase):
    def test_paired_scale(self):
        x=summarize([[(2,1),(4,2)],[(6,3),(8,4)]],100)
        self.assertEqual(x['speedup'],2);self.assertEqual(x['within_session_bootstrap_95'],[2,2])
    def test_null(self):
        x=summarize([[(3,3),(4,4),(5,5)]],100)
        self.assertEqual(x['within_session_bootstrap_95'],[1,1])
    def test_followup_contexts_and_counter_subset(self):
        p=make_plan([1536,1792,2048,2304,2560]);self.assertEqual(sum(c['kind']=='model' for c in p['cases']),15)
        self.assertTrue(any(j['case']=='B_n1792_p0' for j in p['profiles']))
        self.assertFalse(any(j['case']=='B_n2560_p0' for j in p['profiles']))
        self.assertNotIn('torch_compile',{j['backend'] for j in p['profiles']})
    def test_any_context_has_model_preflight(self):
        p=make_plan([512]);self.assertTrue(any(j['case']=='B_n512_p0' for j in p['profiles']))
    def test_reject_unvalidated_context(self):
        with self.assertRaises(ValueError):make_plan([8192])
if __name__=='__main__':unittest.main(verbosity=2)
