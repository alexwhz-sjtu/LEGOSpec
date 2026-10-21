"""QS pulls only allowlisted files using an expiring, forced-command SSH key."""
from pathlib import Path
import argparse,json,subprocess,hashlib,datetime,time,fcntl,os
B=Path('/data/temp/LiteSpec_exp/tests/qs_only_migration_20261019')
D=Path('/mnt/bd-1/task/339953/artifacts/LiteSpec/migrations/Inf_H800_2_20261019')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['legacy','final']);parser.add_argument('manifest');args=parser.parse_args()
 with (B/('direct_'+args.mode+'.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  record=json.loads(Path(args.manifest).read_text());entries=record['files'];verified=[]
  for entry in entries:
   if args.mode=='legacy':source='/data/wanghanzhen/projects/SpecDecoding/'+entry['relative'];target=D/entry['relative']
   else:source=entry['source'];target=D/'current_rnn_ce'/entry['relative']
   target.parent.mkdir(parents=True,exist_ok=True)
   if not(target.is_file() and target.stat().st_size==entry['size'] and digest(target)==entry['sha256']):
    command=['ssh','-C','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(B/'known_hosts'),'-i',str(B/'transfer_ed25519'),'-p','56780','root@223.76.225.211','get '+source]
    child=subprocess.Popen(command,stdout=subprocess.PIPE);temporary=target.with_suffix(target.suffix+'.direct.partial');h=hashlib.sha256();n=0;last=0
    with temporary.open('wb') as f:
     while True:
      chunk=child.stdout.read(8*1024*1024)
      if not chunk:break
      f.write(chunk);h.update(chunk);n+=len(chunk)
      if time.monotonic()-last>10:
       progress={'status':'copying','pid':os.getpid(),'current':str(target),'current_bytes':n,'verified_files':verified}
       (B/('direct_'+args.mode+'_progress.json')).write_text(json.dumps(progress,indent=2)+'\n');last=time.monotonic()
    assert child.wait()==0,'Read-only SSH transfer failed';assert n==entry['size'] and h.hexdigest()==entry['sha256'],'Transfer digest mismatch';temporary.replace(target)
   verified.append({'path':str(target),'size':entry['size'],'sha256':entry['sha256']});print('VERIFIED',str(target),flush=True)
  result={**record,'status':'completed','verified_files':verified,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'transport':'Server-to-server SSH with temporary forced-command read-only allowlist'}
  if args.mode=='final':
   result.update(files=verified)
   (B.parent/'staged_component_search_qs_20261019/imported_rnn_ce_verified.json').write_text(json.dumps(result,indent=2)+'\n')
   (D/'current_rnn_ce/import_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
  (B/('direct_'+args.mode+'_completed.json')).write_text(json.dumps(result,indent=2)+'\n')
  (D/('direct_'+args.mode+'_completed.json')).write_text(json.dumps(result,indent=2)+'\n')
  print('DIRECT_'+args.mode.upper()+'_COMPLETED',flush=True)
if __name__=='__main__':main()
