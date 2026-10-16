"""Re-express dense solves with symbolic QR so native factors expand to GPU code.

This changes graph representation only. check_factors must compare the result
against the independently built native QR factors before rollouts are allowed.
"""
import bootstrap
import argparse,json
from pathlib import Path
import casadi as ca
from optimization.native_export import export
p=argparse.ArgumentParser();p.add_argument('spec',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
original=ca.solve
def symbolic_solve(matrix,rhs,*args,**kwargs):
    return original(matrix,rhs,'symbolicqr') if not args and not kwargs else original(matrix,rhs,*args,**kwargs)
ca.solve=symbolic_solve
try:export(json.loads(a.spec.read_text()),a.output)
finally:ca.solve=original
