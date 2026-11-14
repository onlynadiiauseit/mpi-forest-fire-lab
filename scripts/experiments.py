"""Validate every result before benchmarking; retain raw commands and output."""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import platform
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'measurements'
RAW = OUT / 'raw_runs_new.jsonl'
FIELDS = ['дослід','версія','метод','N','P','d','seed','повтор','час_с','обмін_с','обчислення_с','редукція_с','кроки','згоріло','досягнуто','hash','байти','статус','примітка','потоки','ресурс','команда']
FIELDS += ['облік_с','інше_с','дисбаланс_с','wall_с','профілювання_с','такти']
KEYS = dict(zip(FIELDS[8:16],['час','обмін','обчислення','редукція','кроки','згоріло','досягнуто','hash']))
KEYS.update({'облік_с':'облік','інше_с':'інше','дисбаланс_с':'дисбаланс','wall_с':'wall','профілювання_с':'профілювання','такти':'такти'})
PHYSICAL_CORES = 1


def execute(p, executable, args, threads=1, core=False, timeout=300):
    core = core or (threads == 1 and p <= PHYSICAL_CORES)
    placement = ['--map-by',f'slot:PE={threads}','--bind-to','core'] if core else ['--use-hwthread-cpus','--map-by','hwthread','--bind-to','hwthread']
    cmd = ['mpirun',*placement,'-np',str(p),f'./{executable}',*map(str,args)]
    env = {**os.environ,'OMP_NUM_THREADS':str(threads),'OMP_PROC_BIND':'true','OMP_PLACES':'cores','OMPI_ALLOW_RUN_AS_ROOT':'1','OMPI_ALLOW_RUN_AS_ROOT_CONFIRM':'1'}
    started=time.monotonic()
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    status='успішно'
    try:
        stdout,stderr=proc.communicate(timeout=timeout)
        if proc.returncode: status='помилка'
    except subprocess.TimeoutExpired:
        status='тайм-аут'
        os.killpg(proc.pid,signal.SIGTERM)
        try: stdout,stderr=proc.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            stdout,stderr=proc.communicate()
    raw={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'threads':threads,'binding':'core' if core else 'hwthread','status':status,'returncode':proc.returncode,'wall_s':time.monotonic()-started,'stdout':stdout,'stderr':stderr}
    with RAW.open('a',encoding='utf-8') as f: f.write(json.dumps(raw,ensure_ascii=False)+'\n')
    data=dict(re.findall(r'([\w_]+)=([^\s]+)',stdout))
    result={k:data.get(v,'') for k,v in KEYS.items()}
    result.update({'статус':status,'потоки':threads,'ресурс':f'{p*threads} physical cores' if core else f'{p} logical CPUs','команда':shlex.join(cmd),'примітка':stderr.strip()[-500:]})
    return result,stdout


class Experiment:
    def __init__(self,args):
        self.args=args
        self.processes = sorted(set([p for p in [1,2,4,8,16,32] if p <= args.max_p] + [args.max_p]))
        existing=list(csv.DictReader((ROOT/'results_new.csv').open(encoding='utf-8-sig'))) if args.resume else []
        self.cache={self.key(row):row for row in existing}
        self.handle=(ROOT/'results_new.csv').open('a' if args.resume else 'w',encoding='utf-8' if args.resume else 'utf-8-sig',newline='')
        self.writer=csv.DictWriter(self.handle,FIELDS)
        if not args.resume: self.writer.writeheader()
        self.count=len(existing)

    @staticmethod
    def key(row):
        return tuple(str(row.get(k,'')) for k in FIELDS[:8]+['байти'])

    def find(self,kind,version,method,n='',p='',d='',seed='',repeat=1,byte_count=''):
        row=dict(zip(FIELDS[:8],[kind,version,method,n,p,d,seed,repeat]))
        row['байти']=byte_count
        return self.cache.get(self.key(row))

    def record(self,kind,version,method,n='',p='',d='',seed='',repeat=1,result=None):
        row=dict.fromkeys(FIELDS,'')
        row.update(dict(zip(FIELDS[:8],[kind,version,method,n,p,d,seed,repeat])))
        row.update(result or {})
        if self.key(row) in self.cache: return self.cache[self.key(row)]
        self.writer.writerow(row)
        self.handle.flush()
        self.cache[self.key(row)]=row
        self.count+=1
        return row

    def fire(self,kind,n,p,method='ordered',repeat=1,seed=42,density=.6,path=None,threads=1,core=False,expected=None):
        version='V0' if method=='v0' else '2D' if method=='2d' else 'V1/V3'
        saved=self.find(kind,version,method,n,p,density,seed,repeat)
        if saved is not None: return saved
        options=[method,n,density,seed] if method in ['v0','2d'] else ['stripe',n,density,seed,method]
        if path:
            if path.exists(): path.unlink()
            options.append(path)
        result,_=execute(p,'fire',options,threads,core,self.args.timeout)
        if result['статус']!='успішно' or not result['час_с'] or not result['hash']:
            self.record(kind,version,method,n,p,density,seed,repeat,result)
            raise RuntimeError(f'Run failed: {kind} P={p} {method}: {result}')
        parts = [float(result[k]) for k in ['обмін_с','обчислення_с','редукція_с','облік_с','інше_с']]
        if min(parts) < -1e-8 or not math.isclose(sum(parts),float(result['час_с']),rel_tol=1e-7,abs_tol=1e-7):
            raise RuntimeError(f'Inconsistent critical-step timing profile: {result}')
        if int(result['такти']) != int(result['кроки']) + 1:
            raise RuntimeError('Missing final termination-check timing')
        if not 0<=float(result['дисбаланс_с'])<=float(result['час_с'])+1e-7 or min(float(result[k]) for k in ['wall_с','профілювання_с'])<0:
            raise RuntimeError('Invalid profiling diagnostic')
        if expected:
            for key in ['кроки','згоріло','досягнуто','hash']:
                if result[key]!=expected[key]: raise RuntimeError(f'{method}, P={p}: mismatch in {key}')
            if path and path.read_bytes()!=expected['_file'].read_bytes(): raise RuntimeError('Full grids differ')
            result['примітка']=('Сітка побайтово та всі підсумкові показники збігаються з V0' if path
                                else 'Усі підсумкові показники та хеш збігаються з V0')
        return self.record(kind,version,method,n,p,density,seed,repeat,result)

    def correctness(self):
        cases=[(64,.6,42,self.processes),(17,1.,7,[2,4,8]),(20,0.,1,[2,4,8]),(31,.59,5,[2,4,8]),(8,1.,9,[8]),(1,1.,0,[1]),(1,0.,0,[1])]
        for n,d,seed,ps in cases:
            path=OUT/f'reference_{n}.bin'
            ref=self.fire('коректність',n,1,'v0',seed=seed,density=d,path=path)
            if d in [0,1]:
                expected_values=[n,n*n,1] if d==1 else [0,0,0]
                assert [int(ref[k]) for k in ['кроки','згоріло','досягнуто']]==expected_values
            if n==64 and d==.6 and seed==42:
                assert ref['hash']=='87adc059ebcdb82d' and ref['кроки']=='38' and ref['згоріло']=='628'
            ref['_file']=path
            for p in [p for p in ps if p<=self.args.max_p]:
                for method in ['ordered','nonblocking','overlap','hybrid','2d']:
                    self.fire('коректність',n,p,method,seed=seed,density=d,path=OUT/f'check_{n}_{p}_{method}.bin',threads=2 if method=='hybrid' else 1,core=method=='hybrid' and p*2<=PHYSICAL_CORES,expected=ref)
            print(f'Correctness passed: N={n}, d={d}',flush=True)

    def calibrate(self):
        n=self.args.n or 1024
        for trial in range(5):
            row=self.fire('калібрування',n,1,'v0',repeat=trial+1)
            elapsed=float(row['час_с'])
            print(f'Calibration: N={n}, V0={elapsed:.3f} s',flush=True)
            if 24<=elapsed<=45: return n
            n=max(128,round(n*(28/max(elapsed,.001))**(1/3)/64)*64)
        raise RuntimeError('No V0 configuration in 20-60 s')

    def timings(self,n):
        reference=None
        for p in self.processes:
            for method in (['v0','ordered'] if p==1 else ['ordered']):
                for r in range(self.args.repeats+1):
                    row=self.fire('прогрів_сильне' if r==0 else 'сильне',n,p,method,repeat=r,expected=reference)
                    if reference is None: reference=row
                    print(f'Strong N={n} P={p} {method} r={r}: {row["час_с"]} s',flush=True)
        for p in self.processes:
            wn=round(self.args.weak_n*math.sqrt(p))
            weak_reference=None
            for r in range(self.args.repeats+1):
                seq=self.fire('прогрів_слабке_V0' if r==0 else 'слабке_V0',wn,1,'v0',repeat=r,expected=weak_reference)
                weak_reference=seq
                row=self.fire('прогрів_слабке' if r==0 else 'слабке',wn,p,repeat=r,expected=weak_reference)
                print(f'Weak N={wn} P={p} r={r}: {row["час_с"]} s',flush=True)
        extra_reference=self.fire('еталон_додаткові',self.args.extra_n,1,'v0')
        for p in [p for p in self.processes if p>1]:
            for method in ['ordered','nonblocking','overlap','2d']:
                for r in range(self.args.repeats+1): self.fire('прогрів_додаткові' if r==0 else 'додаткові',self.args.extra_n,p,method,repeat=r,expected=extra_reference)
                print(f'Extra N={self.args.extra_n} P={p} {method}: complete',flush=True)
        for cores in [c for c in [2,4,6] if c<=PHYSICAL_CORES]:
            for p,threads,method in [(cores,1,'nonblocking'),(cores//2,2,'hybrid')]:
                for r in range(self.args.repeats+1): self.fire('прогрів_гібрид' if r==0 else 'гібрид',self.args.extra_n,p,method,repeat=r,threads=threads,core=True,expected=extra_reference)
                print(f'Hybrid comparison: {cores} physical cores, P={p}, threads={threads}',flush=True)

    def deadlock(self):
        def probe(size,repeat=1):
            saved=self.find('поріг_тупика','V2','bad',p=2,repeat=repeat,byte_count=size)
            if saved is not None: return saved['статус']=='тайм-аут'
            result,_=execute(2,'deadlock',[size,100,'bad'],timeout=3)
            result['байти']=size
            self.record('поріг_тупика','V2','bad',p=2,repeat=repeat,result=result)
            if result['статус']=='помилка': raise RuntimeError(result)
            return result['статус']=='тайм-аут'
        low,high=0,None
        for size in [8,64,256,512,1024,2048,4096,8192,16384,32768,65536,1048576,4194304]:
            if probe(size):
                high=size
                break
            low=size
        if high is None: raise RuntimeError('No deadlock up to 4 MiB')
        while high-low>1:
            mid=(high+low)//2
            if probe(mid): high=mid
            else: low=mid
        for r in [2,3]:
            if probe(low,r) or not probe(high,r): raise RuntimeError('Unstable deadlock threshold')
        (OUT/'deadlock_threshold_new.json').write_text(json.dumps({'largest_safe':low,'smallest_deadlock':high,'timeout_s':3,'confirmation_repeats':3}),encoding='utf-8')
        print(f'Deadlock threshold: {low} safe, {high} blocks (bytes)',flush=True)
        if self.find('тупик_автомат','V2','bad',high,2,.6,42,byte_count=high) is None:
            result,_=execute(2,'fire',['stripe',high,.6,42,'bad'],timeout=5)
            if result['статус']!='тайм-аут': raise RuntimeError('Automaton did not deadlock')
            self.record('тупик_автомат','V2','bad',high,2,.6,42,result={**result,'байти':high,'примітка':'Зависання send/send/recv/recv у моделі; завершено через 5 с'})
        for size in sorted(set([low,high,8192,65536,1048576,4194304])):
            for method in ['ordered','nonblocking']:
                for r in range(self.args.repeats+1):
                    if self.find('прогрів_тупик' if r==0 else 'тупик','V2',method,p=2,repeat=r,byte_count=size) is not None: continue
                    result,_=execute(2,'deadlock',[size,1000,method])
                    if result['статус']!='успішно': raise RuntimeError(result)
                    result['байти']=size
                    self.record('прогрів_тупик' if r==0 else 'тупик','V2',method,p=2,repeat=r,result=result)

    def channel(self):
        for r in range(1,self.args.repeats+1):
            if all(self.find('канал','канал','sendrecv',p=2,repeat=r,byte_count=8*2**i) is not None for i in range(20)): continue
            result,stdout=execute(2,'channel',[1000])
            records=re.findall(r'байти=(\d+) повтори=\d+ час=([\d.]+)',stdout)
            if result['статус']!='успішно' or len(records)!=20: raise RuntimeError(result)
            for size,secs in records: self.record('канал','канал','sendrecv',p=2,repeat=r,result={**result,'байти':size,'час_с':secs})
        for n in [256,512,1024]:
            for p in [p for p in [2,4,8] if p<=self.args.max_p]:
                for r in range(self.args.repeats+1): self.fire('прогрів_хокні' if r==0 else 'хокні',n,p,repeat=r)
        print('Channel and Hockney measurements complete',flush=True)

    def finish(self):
        original=list(csv.DictReader((OUT/'percolation_original.csv').open(encoding='utf-8-sig')))
        assert len(original)==1050
        for row in original: self.record('перколяція',row['версія'],row['метод'],row['N'],row['P'],row['d'],row['seed'],row['повтор'],{k:row[k] for k in FIELDS[8:19]})
        self.handle.close()
        (ROOT/'results_new.csv').replace(ROOT/'results.csv')
        RAW.replace(OUT/'raw_runs.jsonl')
        (OUT/'environment_new.json').replace(OUT/'environment.json')
        (OUT/'deadlock_threshold_new.json').replace(OUT/'deadlock_threshold.json')
        print('Results committed:',self.count,'rows',flush=True)


def main():
    global PHYSICAL_CORES
    parser=argparse.ArgumentParser()
    parser.add_argument('--n',type=int,default=0)
    parser.add_argument('--weak-n',type=int,default=384)
    parser.add_argument('--extra-n',type=int,default=512)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--timeout',type=int,default=300)
    parser.add_argument('--max-p',type=int,default=min(32,len(os.sched_getaffinity(0))))
    parser.add_argument('--resume',action='store_true',help='continue the same saved series without repeating completed measurements')
    args=parser.parse_args()
    if args.repeats<3 or args.max_p<2 or args.max_p>len(os.sched_getaffinity(0)) or args.n<0 or args.n>46340 or min(args.weak_n,args.extra_n)<args.max_p or args.timeout<10:
        parser.error('Use repeats>=3, available max-p>=2, valid dimensions and timeout>=10')
    OUT.mkdir(exist_ok=True)
    topology=set()
    for cpu in os.sched_getaffinity(0):
        base=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        topology.add(((base/'physical_package_id').read_text().strip(),(base/'core_id').read_text().strip()))
    PHYSICAL_CORES=len(topology)
    names=['src/fire.c','src/profile.h','src/channel.c','src/deadlock.c']
    if args.resume:
        meta=json.loads((OUT/'environment_new.json').read_text(encoding='utf-8'))
        if any(meta['settings'][k]!=v for k,v in vars(args).items() if k!='resume'):
            parser.error('Resume requires the original measurement settings')
        if any(meta[name+'_sha256']!=hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names):
            parser.error('Source code changed; previous measurements cannot be resumed')
        if PHYSICAL_CORES!=meta['physical_cores'] or sorted(os.sched_getaffinity(0))!=meta['affinity']:
            parser.error('Processor configuration changed')
        if subprocess.check_output(['ompi_info','--version'],text=True)!=meta['mpi'] or subprocess.check_output(['mpicc','--version'],text=True)!=meta['compiler']:
            parser.error('MPI or compiler version changed')
        meta['resumed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    else:
        RAW.write_text('',encoding='utf-8')
        meta={'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'platform':platform.platform(),'os_release':Path('/etc/os-release').read_text(),'cpuinfo':Path('/proc/cpuinfo').read_text(),'meminfo':Path('/proc/meminfo').read_text(),'affinity':sorted(os.sched_getaffinity(0)),'mpi':subprocess.check_output(['ompi_info','--version'],text=True),'compiler':subprocess.check_output(['mpicc','--version'],text=True),'settings':vars(args)}
        meta['physical_cores']=PHYSICAL_CORES
        meta['timing_method']='sum of per-step MPI_MAXLOC durations; components from the same critical rank; profiling collectives excluded'
        for name in names: meta[name+'_sha256']=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    (OUT/'environment_new.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    exp=Experiment(args)
    exp.correctness()
    exp.deadlock()
    n=meta['strong_n'] if args.resume and 'strong_n' in meta else exp.calibrate()
    meta['strong_n']=n
    (OUT/'environment_new.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    exp.timings(n)
    exp.channel()
    exp.finish()


if __name__=='__main__': main()
