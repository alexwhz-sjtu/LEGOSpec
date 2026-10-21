"""Durable PID/start_ticks and known return-code receipt, independent of SSH."""
import json,os,subprocess,sys
from pathlib import Path

state=json.loads(Path(sys.argv[1]).read_text());job=json.loads(Path(sys.argv[2]).read_text())
run=Path(job['run_dir'])
env=os.environ.copy();env.update(job.get('env',{}))
with Path(job['log']).open('ab',buffering=0) as log:
    child=subprocess.Popen(['bash','scripts/composable/run_training_composable.sh',job['config']],
        cwd=state['snapshot'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True)
ticks=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
receipt={'pid':child.pid,'start_ticks':ticks,'watcher_pid':os.getpid()}
temp=run/'launch_receipt.tmp';temp.write_text(json.dumps(receipt));temp.replace(run/'launch_receipt.json')
code=child.wait();receipt['returncode']=code
temp=run/'exit_receipt.tmp';temp.write_text(json.dumps(receipt));temp.replace(run/'exit_receipt.json')
raise SystemExit(code)
