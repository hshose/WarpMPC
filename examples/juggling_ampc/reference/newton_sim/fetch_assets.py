"""Fetch the official Panda meshes, pinning and hashing their upstream revision."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    dest=Path('models/panda/meshes');dest.mkdir(parents=True,exist_ok=True)
    repo='frankarobotics/franka_ros'
    commit=json.load(urllib.request.urlopen(f'https://api.github.com/repos/{repo}/commits/develop'))['sha']
    base=f'https://raw.githubusercontent.com/{repo}/{commit}/franka_description/meshes'
    def fetch(item):
        kind,k,ext=item;name=f'{kind}/link{k}.{ext}';url=base+'/'+name
        data=urllib.request.urlopen(url).read();target=dest/name;target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
        return name,dict(url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    items=[(kind,k,ext) for kind,ext in [('collision','stl'),('visual','dae')] for k in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:files=dict(pool.map(fetch,items))
    (dest/'provenance.json').write_text(json.dumps(dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),
        revision=commit,license='Apache-2.0; ../LICENSE',files=files),indent=2)+'\n')
    print(commit,len(files),'meshes',sum(v['bytes'] for v in files.values()),'bytes')

if __name__=='__main__':main()
