"""Reproduce numerical tables and model estimates from the measured CSV.

This module uses only the standard library and never creates documents.
"""
import csv
import hashlib
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = {'compute':'обчислення_с','exchange':'обмін_с','count':'облік_с',
              'reduction':'редукція_с','other':'інше_с'}


def analyze(root=ROOT):
    rows=list(csv.DictReader((root/'results.csv').open(encoding='utf-8-sig')))
    meta=json.loads((root/'measurements/environment.json').read_text(encoding='utf-8'))
    threshold=json.loads((root/'measurements/deadlock_threshold.json').read_text(encoding='utf-8'))
    for name in ['src/fire.c','src/profile.h','src/channel.c','src/deadlock.c']:
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=meta[name+'_sha256']:
            raise ValueError(f'Measurements refer to a different source revision: {name}')

    def select(kind,method=None,p=None,n=None,cores=None):
        return [r for r in rows if r['дослід']==kind and r['статус']=='успішно'
                and (method is None or r['метод']==method)
                and (p is None or int(r['P'])==p)
                and (n is None or int(r['N'])==n)
                and (cores is None or int(r['P'])*int(r['потоки'])==cores)]

    def med(items,key='час_с'):
        if not items: raise ValueError('A required measurement group is missing')
        return stats.median(float(r[key]) for r in items)

    measured=[r for r in rows if r['дослід']!='перколяція' and r['версія'] in ['V0','V1/V3','2D'] and r['статус']=='успішно']
    for r in measured:
        total=float(r['час_с'])
        parts=[float(r[key]) for key in COMPONENTS.values()]
        if min(parts)<-1e-8 or not math.isclose(sum(parts),total,rel_tol=1e-7,abs_tol=1e-7):
            raise ValueError('Critical-rank components do not sum to the maximum-step time')
        if int(r['такти'])!=int(r['кроки'])+1:
            raise ValueError('The final termination check was not profiled')
        if not 0<=float(r['дисбаланс_с'])<=total+1e-7:
            raise ValueError('Invalid inter-process timing spread')
    failures=[r for r in rows if r['статус']!='успішно' and r['дослід'] not in ['поріг_тупика','тупик_автомат']]
    if failures: raise ValueError('Unexpected failed MPI runs are present')

    ps=sorted({int(r['P']) for r in select('сильне','ordered')})
    t0=med(select('сильне','v0'))
    if not 20<=t0<=60: raise ValueError('The sequential baseline is outside 20-60 seconds')
    strong=[]
    for p in ps:
        group=select('сильне','ordered',p)
        if len(group)<3: raise ValueError('Fewer than three repetitions')
        t=med(group)
        # A single representative measured run keeps the component table additive.
        representative=sorted(group,key=lambda r:float(r['час_с']))[len(group)//2]
        strong.append({'p':p,'n':int(group[0]['N']),'time':t,'min':min(float(r['час_с']) for r in group),
                       'max':max(float(r['час_с']) for r in group),'speed':t0/t,'efficiency':t0/t/p,
                       'steps':int(group[0]['кроки']),'ticks':int(group[0]['такти']),
                       'parts':{k:float(representative[v]) for k,v in COMPONENTS.items()},
                       'spread':float(representative['дисбаланс_с']),'wall':float(representative['wall_с']),
                       'profiling':float(representative['профілювання_с'])})
    reference=select('сильне','v0')[0]
    for r in select('сильне'):
        if any(r[key]!=reference[key] for key in ['кроки','згоріло','досягнуто','hash']):
            raise ValueError('Strong-scaling results differ from V0')

    weak=[]
    weak_base=med(select('слабке','ordered',1))
    for p in ps:
        group=select('слабке','ordered',p)
        n=int(group[0]['N'])
        baseline=select('слабке_V0','v0',1,n)
        for r in group:
            if any(r[k]!=baseline[0][k] for k in ['кроки','згоріло','досягнуто','hash']):
                raise ValueError('Weak-scaling results differ from the same-size V0')
        t,seq=med(group),med(baseline)
        weak.append({'p':p,'n':n,'local_cells':n*n/p,'steps':int(group[0]['кроки']),
                     'time':t,'v0_time':seq,'scaled_speed':seq/t,'time_ratio':t/weak_base,
                     'step_ms':1000*t/int(group[0]['кроки']),
                     'effective_serial':(p-seq/t)/(p-1) if p>1 else None})

    channel=defaultdict(list)
    for r in select('канал'): channel[int(r['байти'])].append(float(r['час_с']))
    if len(channel)!=20 or min(channel)!=8 or max(channel)!=4194304:
        raise ValueError('Channel measurements do not cover the required range')
    alpha=stats.median(float(r['час_с']) for r in select('канал') if int(r['байти'])<=64)
    large=sorted(m for m in channel if m>=65536)
    fit=stats.linear_regression(large,[stats.median(channel[m]) for m in large])
    if fit.slope<=0: raise ValueError('Invalid channel slope')
    beta=1/fit.slope
    ys=[stats.median(channel[m]) for m in large]
    residual=sum((y-(fit.intercept+fit.slope*m))**2 for m,y in zip(large,ys))
    r2=1-residual/sum((y-stats.mean(ys))**2 for y in ys)
    channel_table=[{'bytes':m,'time':stats.median(channel[m]),'min':min(channel[m]),'max':max(channel[m]),
                    'model':alpha+m/beta} for m in sorted(channel)]

    hockney=[]
    for n,p in sorted({(int(r['N']),int(r['P'])) for r in select('хокні')}):
        group=select('хокні','ordered',p,n)
        measured_time=stats.median(float(r['обмін_с'])/int(r['кроки']) for r in group)
        predicted=2*(alpha+n/beta)
        hockney.append({'n':n,'p':p,'predicted':predicted,'measured':measured_time,'ratio':measured_time/predicted})

    extra=[]
    for p in sorted({int(r['P']) for r in select('додаткові')}):
        record={'p':p}
        for method in ['ordered','nonblocking','overlap','2d']:
            record[method]=med(select('додаткові',method,p))
        record['overlap_speed']=record['nonblocking']/record['overlap']
        record['two_d_speed']=record['ordered']/record['2d']
        extra.append(record)
    hybrid=[]
    for cores in sorted({int(r['P'])*int(r['потоки']) for r in select('гібрид')}):
        a,b=select('гібрид','nonblocking',cores=cores),select('гібрид','hybrid',cores=cores)
        if len(a)<3 or len(b)<3 or not all('--bind-to core' in r['команда'] for r in a+b):
            raise ValueError('Hybrid comparison does not use equally bound physical cores')
        hybrid.append({'cores':cores,'mpi_p':cores,'hybrid_p':cores//2,'threads':2,
                       'mpi':med(a),'hybrid':med(b),'speed':med(a)/med(b)})
    deadlock=[]
    for size in sorted({int(r['байти']) for r in select('тупик')}):
        a=med([r for r in select('тупик','ordered') if int(r['байти'])==size])
        b=med([r for r in select('тупик','nonblocking') if int(r['байти'])==size])
        deadlock.append({'bytes':size,'ordered':a,'nonblocking':b,'speed':a/b})
    for size,status in [(threshold['largest_safe'],'успішно'),(threshold['smallest_deadlock'],'тайм-аут')]:
        if sum(r['дослід']=='поріг_тупика' and int(r['байти'])==size and r['статус']==status for r in rows)<3:
            raise ValueError('Deadlock threshold lacks repeated confirmation')
    if not any(r['дослід']=='тупик_автомат' and r['статус']=='тайм-аут' for r in rows):
        raise ValueError('The prescribed automaton deadlock was not reproduced')

    percolation=[]
    groups=defaultdict(list)
    for r in select('перколяція'): groups[float(r['d'])].append(int(r['досягнуто']))
    if len(groups)!=21 or any(len(v)<50 for v in groups.values()): raise ValueError('Incomplete percolation sample')
    for d,values in sorted(groups.items()):
        p,n=stats.mean(values),len(values)
        z=1.96
        center=(p+z*z/(2*n))/(1+z*z/n)
        half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        percolation.append({'d':d,'reached':sum(values),'samples':n,'probability':p,
                            'low':max(0.,min(p,center-half)),'high':min(1.,max(p,center+half))})
    transition=next(a['d']+(.5-a['probability'])*(b['d']-a['d'])/(b['probability']-a['probability'])
                    for a,b in zip(percolation,percolation[1:]) if a['probability']<=.5<=b['probability'] and a['probability']!=b['probability'])

    physical=meta['physical_cores']
    am_points=[r for r in strong if 1<r['p']<=physical]
    am_raw=sum((1-1/r['p'])*(r['time']/strong[0]['time']-1/r['p']) for r in am_points)/sum((1-1/r['p'])**2 for r in am_points)
    gu_points=[r for r in weak if 1<r['p']<=physical]
    gu_raw=sum((r['p']-1)*(r['p']-r['scaled_speed']) for r in gu_points)/sum((r['p']-1)**2 for r in gu_points)
    models=[]
    for r,w in zip(strong,weak):
        p=r['p']
        models.append({'p':p,'strong_relative':strong[0]['time']/r['time'],
                       'amdahl':1/(am_raw+(1-am_raw)/p) if 0<=am_raw<=1 else None,
                       'scaled_measured':w['scaled_speed'],
                       'gustafson':gu_raw+p*(1-gu_raw) if 0<=gu_raw<=1 else None})
    return {'metadata':meta,'row_count':len(rows),'correctness_runs':len(select('коректність')),
            'comparisons':sum(r['дослід']=='коректність' and 'побайтово' in r['примітка'] for r in rows),
            'v0_time':t0,'best':min(strong,key=lambda r:r['time']),'strong':strong,'weak':weak,
            'channel':channel_table,'alpha':alpha,'beta':beta,'channel_r2':r2,'hockney':hockney,
            'extra':extra,'hybrid':hybrid,'deadlock':deadlock,'threshold':threshold,
            'percolation':percolation,'percolation_threshold':transition,'models':models,
            'amdahl_s':am_raw,'gustafson_s':gu_raw,'model_fit_p':[r['p'] for r in am_points]}


def write_outputs(data,root=ROOT):
    folder=root/'measurements'
    (folder/'analysis.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for name in ['strong','weak','channel','hockney','extra','hybrid','deadlock','percolation','models']:
        records=[]
        for row in data[name]:
            flattened={}
            for key,value in row.items():
                if isinstance(value,dict): flattened.update({f'{key}_{k}':v for k,v in value.items()})
                else: flattened[key]=value
            records.append(flattened)
        with (folder/f'analysis_{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


if __name__=='__main__':
    result=analyze()
    write_outputs(result)
    print(f'Validated {result["row_count"]} records, {result["comparisons"]} full-grid comparisons.')
    print(f'V0: {result["v0_time"]:.3f} s; alpha: {result["alpha"]*1e6:.3f} us; beta: {result["beta"]/1e9:.3f} GB/s')
    print('Reproducible tables: measurements/analysis_*.csv')
