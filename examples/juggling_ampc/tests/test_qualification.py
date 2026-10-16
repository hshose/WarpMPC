import contextlib, io, json, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qualify_expert import qualify, qualifies


class QualificationTests(unittest.TestCase):
    def fixture(self,root,uptime=10.,contacts=10):
        native=dict(seeds=[str(i) for i in range(128)],uptime_s=[uptime]*128,
                    contacts_before_loss=[contacts]*128,policy_kind='native_diagnostic',
                    calibration_sha256='fit',noise_disturbance_multiplier=1.5,drop_bounds={'z_min':1.,'z_max':1.5},rollout_seconds=30)
        warp=dict(native,policy_kind='warp_mpc',mean_uptime_s=uptime)
        for name,data in [('native',native),('warp',warp),('parity',{'passed':True}),('inflight_parity',{'passed':True,'cases':[{'passed':True}]*16}),('metadata',{'spec':{'sqp_iters':2}})]:
            (root/f'{name}.json').write_text(json.dumps(data))
        return SimpleNamespace(**{k:root/f'{k}.json' for k in ('native','warp','parity','inflight_parity','metadata','output')},minimum_ratio=.9)

    def test_native_relative_repeat_juggling_without_false_survival_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory))
            with contextlib.redirect_stdout(io.StringIO()):result=qualify(args)
            self.assertTrue(result['passed']);self.assertEqual(result['warp_survival_30s'],0.)
            report=json.loads(args.warp.read_text())
            self.assertTrue(qualifies(report,result,{'sqp_iters':2}))
            report['noise_disturbance_multiplier']=1.
            with self.assertRaises(ValueError):qualifies(report,result,{'sqp_iters':2})

    def test_arm_multiplier_mismatch_cannot_reuse_qualification(self):
        from uncertainty_settings import settings
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory))
            with contextlib.redirect_stdout(io.StringIO()):certificate=qualify(args)
            report=json.loads(args.warp.read_text())
            report['uncertainty_multipliers']=settings(1.5,1.)
            with self.assertRaises(ValueError):qualifies(report,certificate,{'sqp_iters':2})

    def test_explicit_reuse_does_not_requalify_new_device_or_noise(self):
        from uncertainty_settings import settings
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory))
            with contextlib.redirect_stdout(io.StringIO()):certificate=qualify(args)
            certificate['reuse_completed_gates']=True
            report=json.loads(args.warp.read_text())
            report.update(noise_disturbance_multiplier=1.,uncertainty_multipliers=settings(1.,1.5),implementation_sha256={'rollout.py':'new wrapper'})
            self.assertTrue(qualifies(report,certificate,{'sqp_iters':2}))
            report['mean_uptime_s']=.1
            self.assertFalse(qualifies(report,certificate,{'sqp_iters':2}))

    def test_changed_expert_implementation_requires_requalification(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory));report=json.loads(args.warp.read_text())
            report.update(solver_numerics={'kkt_refinement_steps':0},implementation_sha256={'qp.py':'original'})
            args.warp.write_text(json.dumps(report))
            with contextlib.redirect_stdout(io.StringIO()):certificate=qualify(args)
            self.assertTrue(qualifies(report,certificate,{'sqp_iters':2}))
            for key,value in [('solver_numerics',{'kkt_refinement_steps':1}),('implementation_sha256',{'qp.py':'changed'})]:
                changed=dict(report);changed[key]=value
                with self.assertRaises(ValueError):qualifies(changed,certificate,{'sqp_iters':2})

    def test_matching_immediate_failures_are_not_stabilization(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory),uptime=.6,contacts=1)
            with contextlib.redirect_stdout(io.StringIO()):result=qualify(args)
            self.assertFalse(result['passed'])

    def test_failed_inflight_parity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory));args.inflight_parity.write_text(json.dumps({'passed':False,'cases':[]}))
            with self.assertRaises(ValueError):qualify(args)

    def test_unpaired_seed_sets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory));data=json.loads(args.warp.read_text());data['seeds'][0]='other';args.warp.write_text(json.dumps(data))
            with self.assertRaises(ValueError):qualify(args)


if __name__=='__main__':unittest.main()
