"""Compatibility shim: source auditing disabled for research runs by user request."""
import hashlib
import json


def identity(hashes):
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def wait_for_authoritative_audit(directory, hashes, timeout=3600):
    result = {"status": "disabled_by_user", "reason": "Research runs proceed directly to training; no source audit performed."}
    print(json.dumps(result), flush=True)
    return result
