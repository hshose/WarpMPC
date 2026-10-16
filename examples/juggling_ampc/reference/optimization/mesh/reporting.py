"""Shared arm report with a correct nonuniform-grid title."""
import inspect
from arm import reporting as baseline

source=inspect.getsource(baseline.report).replace(
    "nodes {1000*config['mpc']['h']:g} ms",
    "{config['mpc']['horizon']} nonuniform intervals / {config['mpc']['mesh_horizon']:g} s")
namespace=dict(baseline.__dict__)
exec(compile(source,__file__,'exec'),namespace)
report=namespace['report']
