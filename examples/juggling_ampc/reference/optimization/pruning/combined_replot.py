"""Correct archived combined-run captions using saved traces only."""
import argparse,json,shutil,subprocess
from pathlib import Path
import numpy as np
from optimization.mesh.reporting import report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directories',nargs='+');args=parser.parse_args()
    for path in args.directories:
        directory=Path(path);config=json.loads((directory/'config.json').read_text())
        assert 'mesh_horizon' in config['mpc']
        metrics=json.loads((directory/'metrics.json').read_text())
        report(directory,np.load(directory/'arrays.npz'),np.load(directory/'plant.npz'),config,metrics,None)
        source=directory/'source/optimization/mesh/reporting.py';source.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile('optimization/mesh/reporting.py',source)
        shutil.copyfile(__file__,directory/'source/optimization/pruning/combined_replot.py')
        text=subprocess.check_output(['pdftotext',str(directory/'report.pdf'),'-'],text=True)
        caption=f"{config['mpc']['horizon']} nonuniform intervals / {config['mpc']['mesh_horizon']:g} s"
        assert caption in text and 'nodes 30 ms' not in text
        (directory/'report_caption_correction.json').write_text(json.dumps(dict(
            caption=caption,caption_verified=True,physics_rerun=False,
            source='optimization/mesh/reporting.py',
            note='Regenerated only report.pdf from archived arrays/config/metrics; numerical traces and physical results unchanged.'),indent=2)+'\n')
        print(directory,caption)


if __name__=='__main__':main()
