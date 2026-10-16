import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collect import collect


class CollectionGateTests(unittest.TestCase):
    def test_failed_part_stops_before_next_expensive_rollout(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);part=root/'parts/0000';part.mkdir(parents=True)
            (part/'manifest.json').write_text(json.dumps(dict(trajectories=128,
                rollout=dict(seed=2026,rollout_seconds=30),expert_stabilization_passed=False)))
            with patch('collect.subprocess.run') as launch:
                with self.assertRaisesRegex(RuntimeError,'failed expert stabilization'):
                    collect(SimpleNamespace(output=root,batch=256,microbatch=128,seed=2026),[])
            launch.assert_not_called()
            self.assertFalse((root/'manifest.json').exists())


if __name__=='__main__':unittest.main()
