import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

from benchmarks.vram_cases import vram_case
from benchmarks.monitor_vram import summarize

class VRAMMonitorTests(unittest.TestCase):
    def test_labels_do_not_retain_array_objects_and_nested_scope_restores(self):
        class Array: pass
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'VRAM_CASE_DIR': d}):
            array=Array();ref=weakref.ref(array)
            case=vram_case(512,metadata={'array':array,'dtype':'float32','variant':'baseline'})
            del array
            self.assertIsNone(ref())
            active=Path(d)/f'{os.getpid()}.json'
            with case:
                outer=json.loads(active.read_text())
                with vram_case(1024): self.assertNotEqual(json.loads(active.read_text())['case_id'],outer['case_id'])
                self.assertEqual(json.loads(active.read_text()),outer)
            self.assertFalse(active.exists())

    def test_disabled_has_no_filesystem_side_effect(self):
        with patch.dict(os.environ, {}, clear=True), patch('pathlib.Path.mkdir', side_effect=AssertionError):
            with vram_case(512): pass

    def test_summary_preserves_unsampled_cases(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'cases').mkdir()
            events=[dict(case_id=c,pid=1,event='start',time_ns=i,labels={'batch_size':i}) for i,c in enumerate(['sampled','missed'],1)]
            (out/'cases/1.jsonl').write_text('\n'.join(map(json.dumps,events))+'\n')
            (out/'samples.csv').write_text('time_ns,case_id,gpu_uuid,used_bytes,pids\n1,sampled,GPU-a,100,[1]\n2,sampled,GPU-a,300,[1]\n')
            summarize(out)
            with (out/'case_peaks.csv').open() as f:
                rows={r['case_id']:r for r in csv.DictReader(f)}
            self.assertEqual(rows['sampled']['peak_bytes'],'300')
            self.assertEqual(rows['missed']['peak_bytes'],'')
            self.assertEqual(rows['missed']['samples'],'0')

    def test_subprocess_attribution_and_peak_aggregation(self):
        # Fake only the external NVML provider; exercise real process ancestry,
        # boundary files, subprocess exit propagation and CSV generation.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fake=root/'pynvml.py'
            fake.write_text('''import os,json
from pathlib import Path
from types import SimpleNamespace
NVML_VALUE_NOT_AVAILABLE=2**64-1
class NVMLError(Exception):pass
def nvmlInit():pass
def nvmlShutdown():pass
def nvmlDeviceGetCount():return 1
def nvmlDeviceGetHandleByIndex(i):return i
def nvmlDeviceGetUUID(h):return 'GPU-test'
def nvmlDeviceGetComputeRunningProcesses(h):
    return [SimpleNamespace(pid=json.loads(p.read_text())['pid'],usedGpuMemory=123456) for p in Path(os.environ['TEST_CASES']).glob('*.json') if p.exists()]
''')
            output=root/'result';env=dict(os.environ,PYTHONPATH=str(root)+os.pathsep+str(Path.cwd()),TEST_CASES=str(output/'cases'))
            command=[sys.executable,'benchmarks/monitor_vram.py','--output-dir',str(output),'--interval','.02','--',sys.executable,'-c',
                     "import time; from benchmarks.vram_cases import vram_case\nwith vram_case(123, metadata={'dtype':'float32'}): time.sleep(.2)\nraise SystemExit(3)"]
            result=subprocess.run(command,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,3,result.stderr)
            with (output/'case_peaks.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(rows[0]['batch_size'],'123')
            self.assertEqual(rows[0]['peak_bytes'],'123456')
            self.assertGreater(int(rows[0]['samples']),0)
            self.assertEqual(json.loads((output/'metadata.json').read_text())['returncode'],3)

if __name__=='__main__':unittest.main()
