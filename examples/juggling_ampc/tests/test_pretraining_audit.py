"""Training compatibility must not reinstate the removed remote audit gate."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pretraining_audit import identity, wait_for_authoritative_audit


class AuditTests(unittest.TestCase):
    def test_identity_does_not_depend_on_dictionary_order(self):
        self.assertEqual(identity({'a': '1', 'b': '2'}), identity({'b': '2', 'a': '1'}))
        self.assertNotEqual(identity({'a': '1'}), identity({'a': '2'}))

    def test_training_does_not_wait_for_a_remote_acknowledgement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acknowledgement = json.dumps({'approved': False, 'nonce': 'old'})
            (root / 'ack.json').write_text(acknowledgement)
            result = wait_for_authoritative_audit(root, {'model.json': 'current'}, timeout=0)
            self.assertEqual(result['status'], 'disabled_by_user')
            self.assertFalse((root / 'request.json').exists())
            self.assertEqual((root / 'ack.json').read_text(), acknowledgement)
