import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((root / 'results.csv').open(encoding='utf-8-sig')))
folder = root / 'figures'
folder.mkdir(exist_ok=True)

def groups(kind, method='ordered'):
    a = defaultdict(list)
    for r in rows:
        if r['дослід'] == kind and r['метод'] == method and r['статус'] == 'успішно':
            a[int(r['P'])].append(r)
    return {k: v for k, v in sorted(a.items())}

strong = groups('сильне')
seq = [float(r['час_с']) for r in rows if r['дослід'] == 'сильне' and r['версія'] == 'V0' and r['статус'] == 'успішно']
if strong and seq:
    t0 = float(np.median(seq))
    ps = list(strong)
    speed = [t0 / np.median([float(r['час_с']) for r in strong[p]]) for p in ps]
    plt.figure()
    plt.plot(ps, speed, 'o-', label='Виміряне прискорення')
    plt.plot(ps, ps, '--', label='Ідеальне прискорення')
    plt.xlabel('Кількість процесів P')
    plt.ylabel('Прискорення S(P)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / 'strong_speed.png', dpi=160)
    plt.close()
    plt.figure()
    plt.plot(ps, [s / p for p, s in zip(ps, speed)], 'o-')
    plt.xlabel('Кількість процесів P')
    plt.ylabel('Ефективність E(P)')
    plt.tight_layout()
    plt.savefig(folder / 'strong_efficiency.png', dpi=160)
    plt.close()
    plt.figure()
    plt.plot(ps, [100 * np.median([float(r['обмін_с']) / float(r['час_с']) for r in strong[p]]) for p in ps], 'o-')
    plt.xlabel('Кількість процесів P')
    plt.ylabel('Частка обміну, %')
    plt.tight_layout()
    plt.savefig(folder / 'communication_share.png', dpi=160)
    plt.close()
weak = groups('слабке')
if weak:
    plt.figure()
    plt.plot(list(weak), [np.median([float(r['час_с']) for r in weak[p]]) for p in weak], 'o-')
    plt.xlabel('Кількість процесів P')
    plt.ylabel('Час, с')
    plt.tight_layout()
    plt.savefig(folder / 'weak_scaling.png', dpi=160)
    plt.close()
perc = defaultdict(list)
for r in rows:
    if r['дослід'] == 'перколяція' and r['статус'] == 'успішно':
        perc[float(r['d'])].append(int(r['досягнуто']))
if perc:
    ds = sorted(perc)
    plt.figure()
    plt.plot(ds, [np.mean(perc[d]) for d in ds], 'o-')
    plt.axvline(0.5927, linestyle='--', label='Теоретичний поріг 0,5927')
    plt.xlabel('Щільність лісу d')
    plt.ylabel('Частка досягнення правого краю')
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / 'percolation.png', dpi=160)
    plt.close()
