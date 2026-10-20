"""Record native simulator failures that bypass Python exception handling."""
import json
from pathlib import Path
import re
import sys
import time

code=int(sys.argv[1])
arguments=sys.argv[2:]
if arguments and arguments[0].endswith('record_process_exit.py'):
    sys.exit(0)
if '--run_name' in arguments:
    name=arguments[arguments.index('--run_name')+1]
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',name):
        raise ValueError('Invalid run name')
    directory=Path(__file__).resolve().parent/'logs'/name
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'process_exit_code.txt').write_text(str(code)+'\n')
    status_path=directory/'status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    status['process_exit_code']=code
    if status.get('status') not in ['completed','failed']:
        status.update(status='failed',process_exit_code=code,time=time.time(),
                      error='Process ended before writing complete results; see console log')
    status_path.write_text(json.dumps(status,indent=2))
