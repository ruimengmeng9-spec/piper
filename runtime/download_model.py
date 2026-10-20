"""Fetch only Piper/gripper assets from the pinned official upstream revision."""
import concurrent.futures
import json
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent / 'vendor' / 'agx_arm_urdf'
REV = '983788b58d7eae76511177f768d85877534917fe'
REPO = 'agilexrobotics/agx_arm_urdf'

def fetch(url):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Piper-training-setup'})
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))

def main():
    tree = json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{REV}?recursive=1'))
    entries = [e for e in tree['tree'] if e['type'] == 'blob' and (
        e['path'].startswith(('piper/', 'agx_gripper/')) or e['path'] in ('LICENSE', 'README.md'))]
    ROOT.mkdir(parents=True, exist_ok=True)
    def download(entry):
        dest = ROOT / entry['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_size != entry.get('size'):
            dest.write_bytes(fetch(f'https://raw.githubusercontent.com/{REPO}/{REV}/{entry["path"]}'))
        return entry['path']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for path in pool.map(download, entries):
            print(path, flush=True)
    (ROOT / 'UPSTREAM_REVISION').write_text(REV + '\n', encoding='utf-8')
    print('TOTAL_BYTES', sum(e.get('size', 0) for e in entries))

if __name__ == '__main__':
    main()
