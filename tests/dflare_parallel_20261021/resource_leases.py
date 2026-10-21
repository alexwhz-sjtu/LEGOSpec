"""Reserve only idle GPU UUIDs, sharing the formal scheduler's advisory locks."""
from pathlib import Path
import fcntl,os,socket,sys
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'benchmark'))
from available_gpu_scheduler import available_gpus
LEASES=[]
SELECTED=[]
def free(world_size,port):
    global SELECTED
    try:
        with socket.socket() as s:s.bind(('127.0.0.1',port))
    except OSError:return False
    available=available_gpus();uuids={x['uuid'] for x in available}
    if LEASES:
        if all(x['uuid'] in uuids for x in SELECTED):return True
        for f in LEASES:f.close()
        LEASES.clear();SELECTED=[]
    if len(available)<world_size:return False
    locks=BASE/'benchmark/gpu_locks'
    for gpu in available:
        f=(locks/(gpu['uuid']+'.lock')).open('a')
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:f.close();continue
        if gpu['uuid'] not in {g['uuid'] for g in available_gpus()}:f.close();continue
        LEASES.append(f);SELECTED.append(gpu)
        if len(LEASES)==world_size:break
    if len(LEASES)!=world_size:
        for f in LEASES:f.close()
        LEASES.clear();SELECTED=[];return False
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(g['uuid'] for g in SELECTED)
    return True
