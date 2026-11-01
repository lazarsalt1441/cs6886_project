import csv,io,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import build_plan,backends,parse_ncu,ncu_export_command,check_arrays,write_json,verify_assets
import numpy as np

def csv_text(actions):
    f=io.StringIO();w=csv.writer(f)
    w.writerow(['ID','Process ID','Context','Stream','Metric Name','Metric Unit','Metric Value'])
    for action,metric,unit,value in actions:w.writerow([action,100,1,2,metric,unit,value])
    return '==PROF== replay completed\n'+f.getvalue()

class Counters(unittest.TestCase):
    def parse(self,text,replay='app-range',calls=1):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.csv';p.write_text(text);return parse_ncu(p,calls,replay)
    def test_range_units_normalization(self):
        r=self.parse(csv_text([(0,'dram__bytes_read.sum','Mbyte','3'),(0,'dram__bytes_write.sum','Kbyte','1,500')]),calls=3)
        self.assertEqual(r['read_bytes_per_call'],1e6);self.assertEqual(r['write_bytes_per_call'],500000)
    def test_missing_counter_rejected(self):
        with self.assertRaises(ValueError):self.parse(csv_text([(0,'dram__bytes_read.sum','byte',40)]))
    def test_na_rejected(self):
        with self.assertRaises(ValueError):self.parse(csv_text([(0,'dram__bytes_read.sum','byte','n/a'),(0,'dram__bytes_write.sum','byte',4)]))
    def test_duplicate_not_double_counted(self):
        with self.assertRaises(ValueError):self.parse(csv_text([(0,'dram__bytes_read.sum','byte',4)]*2))
    def test_empty_profile_rejected(self):
        with self.assertRaises(ValueError):self.parse('No kernels were profiled')
    def test_multiple_ranges_rejected(self):
        with self.assertRaises(ValueError):self.parse(csv_text([(i,m,'byte',4) for i in [0,1] for m in ['dram__bytes_read.sum','dram__bytes_write.sum']]))
    def test_kernel_sum_separately_labelled(self):
        r=self.parse(csv_text([(i,m,'byte',4) for i in [0,1] for m in ['dram__bytes_read.sum','dram__bytes_write.sum']]),replay='application')
        self.assertEqual(r['total_bytes_per_call'],16);self.assertIn('excludes copy-only',r['scope'])
    def test_unknown_unit_rejected(self):
        with self.assertRaises(ValueError):self.parse(csv_text([(0,'dram__bytes_read.sum','GiB',4),(0,'dram__bytes_write.sum','GiB',4)]))
    def test_current_ncu_export_options(self):
        cmd=ncu_export_command('ncu','profile.ncu-rep')
        self.assertNotIn('--units',cmd);self.assertEqual(cmd[cmd.index('--print-units')+1],'base')
        self.assertNotIn('--print-metric-name',cmd)
    def test_wide_raw_csv_with_units(self):
        text='"ID","Process ID","Kernel Name","dram__bytes_read.sum","dram__bytes_write.sum"\n"","","","byte","byte"\n"0","818","range","201,326,592","100,663,296"\n'
        result=self.parse(text,calls=3)
        self.assertEqual(result['read_bytes_per_call'],67108864)
        self.assertEqual(result['write_bytes_per_call'],33554432)
        self.assertEqual(result['csv_format'],'wide')
    def test_wide_kernel_rows_summed(self):
        text='ID,Process ID,dram__bytes_read.sum,dram__bytes_write.sum\n,,,byte\n'
        text='ID,Process ID,dram__bytes_read.sum,dram__bytes_write.sum\n,,byte,byte\n0,818,100,200\n1,818,300,400\n'
        result=self.parse(text,replay='application')
        self.assertEqual(result['total_bytes_per_call'],1000)
    def test_wide_requires_units(self):
        text='ID,dram__bytes_read.sum,dram__bytes_write.sum\n0,100,200\n'
        with self.assertRaises(ValueError):self.parse(text)
    def test_wide_duplicate_rejected(self):
        text='ID,dram__bytes_read.sum,dram__bytes_write.sum\n,byte,byte\n0,100,200\n0,100,200\n'
        with self.assertRaises(ValueError):self.parse(text)
    def test_error_after_valid_csv_rejected(self):
        text=csv_text([(0,'dram__bytes_read.sum','byte',4),(0,'dram__bytes_write.sum','byte',4)])+'\n==ERROR== partial data\n'
        with self.assertRaises(ValueError):self.parse(text)

class GatesAndPlanning(unittest.TestCase):
    def test_nsys_copy_payload_not_dram_sum(self):
        import sqlite3
        from analyze import nsys_summary
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'trace.sqlite';db=sqlite3.connect(path)
            db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (copyKind INT, bytes INT)')
            db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?)',[(8,100),(8,200)])
            db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT)');db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?)',[(1,),(2,),(3,)])
            db.commit();db.close();result=nsys_summary(path,3)
            self.assertEqual(result['logical_memcpy_bytes_per_call'],100)
            self.assertEqual(result['kernel_count_per_call'],1)
            self.assertIn('do not add',result['scope'])
    def test_main_profile_prompts_vs_latency_prompts(self):
        p=build_plan('B','main');c={x['id']:x for x in p['cases']}
        self.assertEqual(len([x for x in p['cases'] if x['kind']=='model']),9)
        self.assertTrue(all(c[x['case']]['prompt']==0 for x in p['profiles'] if c[x['case']]['kind']=='model'))
    def test_decode_no_fake_commit(self):
        self.assertEqual(backends(dict(kind='model',set='A')),['iree_stable','torch_compile'])
    def test_extended_has_conditioning_and_three_prompts(self):
        p=build_plan('B','extended');self.assertTrue(any(x['cache']=='evicted' for x in p['profiles']))
        self.assertTrue(any('_p2' in x['case'] for x in p['profiles']))
    def test_exact_prefix_contract(self):
        old=np.arange(8,dtype='float32').reshape(1,1,2,4);new=np.ones((1,1,1,4),dtype='float32')
        case=dict(kind='model',set='A');host=[np.array([0]),np.array([0]),np.array([0]),old]
        refs={'out0':np.array([0.],dtype='float32'),'out1':new}
        actual=[refs['out0'],np.concatenate([old,new],axis=2)]
        self.assertTrue(check_arrays(actual,host,refs,case)['pass_'])
        actual[1][0,0,0,0]+=.000001
        self.assertFalse(check_arrays(actual,host,refs,case)['pass_'])
    def test_forced_winners_all_torch_backends(self):
        from worker import validate_modes
        for name in ['torch_eager','torch_sdpa','torch_compile']:
            tensor=types.SimpleNamespace(to=lambda device:np.array([0.,-100.]))
            fake_torch=types.SimpleNamespace(from_numpy=lambda x:types.SimpleNamespace(to=lambda device:x))
            runner=types.SimpleNamespace(case=dict(kind='model',set='B'),backend=name,inputs=[0,np.array([1,2])],host=[0,np.array([1,2])],refs={},cache=None,device='cpu',inputs_unchanged=lambda:True)
            runner.fn=lambda:runner.inputs[1];runner.validate=lambda result:dict(pass_=True)
            def fixture(cache,case,winner):return [0,np.array([0.,-100.] if winner=='0' else [-100.,0.])],{}
            with patch.dict(sys.modules,{'torch':fake_torch}),patch('worker.load_fixture',fixture):
                self.assertEqual(set(validate_modes(runner)),{'natural','winner_0','winner_1'})
            np.testing.assert_array_equal(runner.inputs[1],[1,2])
    def test_failed_forced_winner_blocks_profile(self):
        from worker import validate_modes
        fake_torch=types.SimpleNamespace(from_numpy=lambda x:types.SimpleNamespace(to=lambda device:x))
        runner=types.SimpleNamespace(case=dict(kind='model',set='B'),backend='torch_compile',inputs=[0,np.array([1.,2.])],host=[0,np.array([1.,2.])],refs={},cache=None,device='cpu',inputs_unchanged=lambda:True)
        runner.fn=lambda:runner.inputs[1];runner.validate=lambda result:dict(pass_=not np.array_equal(result,[-100.,0.]))
        with patch.dict(sys.modules,{'torch':fake_torch}),patch('worker.load_fixture',lambda cache,case,winner:([0,np.array([0.,-100.] if winner=='0' else [-100.,0.])],{})):
            with self.assertRaises(RuntimeError):validate_modes(runner)
    def test_profile_orchestration_builds_real_task(self):
        import study
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache=root/'cache';fixture=cache/'data/stream_control/fixture.json';write_json(fixture,{})
            case=dict(id='stream_control',kind='control',size_mib=64)
            job=dict(case='stream_control',backend='torch_stream',cache='warm',calls=3,replay='app-range',rep=0)
            args=types.SimpleNamespace(cache=cache,ncu='ncu',timeout=10)
            env={k:'' for k in ['torch','cuda','python','gpu','arch','iree','ncu','tf32','model_precision','assets_manifest_sha256','gpu_identity']}
            def fake_run(cmd,log,timeout):
                if '--import' in cmd:
                    self.assertNotIn('--units',cmd)
                    self.assertIn('--print-units',cmd)
                    Path(log).write_text('ID,dram__bytes_read.sum,dram__bytes_write.sum\n,byte,byte\n0,400,400\n')
                else:
                    task=json.loads(Path(cmd[-1]).read_text());write_json(task['result'],dict(valid=True,completed=True));Path(log).write_text('test double')
            with patch('study.run',fake_run):
                result=study.profile(job,case,args,root/'out',env)
                self.assertTrue(result['valid']);self.assertEqual(result['task']['case'],case)
    def test_recover_capture_after_failed_export(self):
        import study
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache=root/'cache';write_json(cache/'data/stream_control/fixture.json',{})
            case=dict(id='stream_control',kind='control',size_mib=64)
            job=dict(case='stream_control',backend='torch_stream',cache='warm',calls=3,replay='app-range',rep=0)
            args=types.SimpleNamespace(cache=cache,ncu='ncu',timeout=10)
            env={k:'' for k in ['torch','cuda','python','gpu','arch','iree','ncu','tf32','model_precision','assets_manifest_sha256','gpu_identity']}
            counts={'capture':0,'export':0}
            def fake_run(cmd,log,timeout):
                if '--import' in cmd:
                    counts['export']+=1
                    if counts['export']==1:raise RuntimeError('Reproduced export failure')
                    Path(log).write_text('ID,dram__bytes_read.sum,dram__bytes_write.sum\n,byte,byte\n0,400,400\n')
                else:
                    counts['capture']+=1;task=json.loads(Path(cmd[-1]).read_text())
                    Path(cmd[cmd.index('-o')+1]+'.ncu-rep').write_bytes(b'fixture report')
                    write_json(task['result'],dict(task=task,valid=True,completed=True));Path(log).write_text('test double')
            with patch('study.run',fake_run):
                self.assertFalse(study.profile(job,case,args,root/'out',env)['valid'])
                recovered=study.profile(job,case,args,root/'out',env)
                self.assertTrue(recovered['valid']);self.assertTrue(recovered['capture_reused'])
            self.assertEqual(counts,{'capture':1,'export':2})
    def test_asset_integrity(self):verify_assets()

if __name__=='__main__':unittest.main()
