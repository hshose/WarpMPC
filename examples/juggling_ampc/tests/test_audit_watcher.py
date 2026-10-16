import contextlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_remote_training import audit


class AuditWatcherTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(ssh_config=Path('/tmp/ssh-config'), host='test-host',
            remote_directory=Path('/tmp/test-audit'), remote_python='python',
            source=Path('/tmp/source'), calibration=Path('/tmp/source/model.json'),
            timeout=30, poll_seconds=0)

    def test_retry_and_old_ack_do_not_consume_new_request(self):
        hashes={'model.json':'current'}
        old=json.dumps(dict(request=dict(nonce='old',live_source_hashes=hashes),ack=dict(nonce='old')))
        fresh=json.dumps(dict(request=dict(nonce='fresh',live_source_hashes=hashes),ack=dict(nonce='old')))
        with patch('audit_remote_training.subprocess.check_output',side_effect=[subprocess.CalledProcessError(255,'ssh'),old,fresh]), \
             patch('audit_remote_training.subprocess.run') as send, \
             patch('audit_remote_training.fingerprint',return_value=hashes) as read, \
             patch('audit_remote_training.time.sleep'),contextlib.redirect_stdout(io.StringIO()):
            result=audit(self.args())
        read.assert_called_once()
        self.assertTrue(result['approved'])
        self.assertEqual(result['nonce'],'fresh')
        self.assertEqual(json.loads(send.call_args.kwargs['input']),result)

    def test_refitted_source_sends_rejection_before_raising(self):
        request=json.dumps(dict(request=dict(nonce='fresh',live_source_hashes={'model.json':'old'}),ack=None))
        with patch('audit_remote_training.subprocess.check_output',return_value=request), \
             patch('audit_remote_training.subprocess.run') as send, \
             patch('audit_remote_training.fingerprint',return_value={'model.json':'refitted'}), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'refitted'):audit(self.args())
        ack=json.loads(send.call_args.kwargs['input'])
        self.assertFalse(ack['approved'])
        self.assertEqual(ack['changed'],['model.json'])


if __name__=='__main__':unittest.main()
