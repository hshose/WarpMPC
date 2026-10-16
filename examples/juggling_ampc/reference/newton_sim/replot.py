"""Regenerate derived PDFs from saved traces without rerunning any physics."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from arm.model import Panda
from arm.reporting import report
from .reporting import contact_report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+',type=Path);args=parser.parse_args()
    for run in args.runs:
        cfg=json.loads((run/'config.json').read_text());m=json.loads((run/'metrics.json').read_text())
        a=np.load(run/'arrays.npz');p=np.load(run/'plant.npz')
        report(run,a,p,cfg,m,Panda());contact_report(run,p,m)
        (run/'report_generation.json').write_text(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),
            source='arm.reporting.report and newton_sim.reporting.contact_report',physics_rerun=False),indent=2)+'\n')
        print(run)

if __name__=='__main__':main()
